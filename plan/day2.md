# Day 2 学习笔记：BlockManager 的分配、回收与 Prefix Cache

> 本日把 `nanovllm/engine/block_manager.py` 从头啃到尾，并和 Day 1 的调度-执行脉络串成一条完整的"块生命周期"。

---

## 0. BlockManager 的定位

一句话：**KV cache 物理池的"房管员"**。它管一个固定大小的"块池"（`num_blocks` 个块、每块 `block_size=256` tokens），对外提供四件事：

1. 给新来的 seq 分块（`allocate`）。
2. decode 时按需补块（`may_append`）。
3. seq 结束/被抢占时归还块（`deallocate`）。
4. 让不同 seq 的相同前缀**共享同一批物理块**（prefix cache）。

真正的 KV tensor 不在这里，它在 `ModelRunner.allocate_kv_cache` 里（那个 `[2, L, num_blocks, 256, heads, dim]` 的大 tensor）。BlockManager 只管**"第 i 号块"这个抽象资源**——块里的 K/V 由 attention 层的 Triton kernel 按 `slot_mapping` 写入。

类比：**操作系统的页框管理器 + 内容去重（CAS）表**。

---

## 1. 核心数据结构

### 1.1 `Block` 类（4 个字段）

| 字段 | 含义 |
|------|------|
| `block_id` | 物理块编号，0..num_blocks-1，**永不变** |
| `ref_count` | 当前有几条 seq 在用这个块（共享引用计数） |
| `hash` | 该块内容的级联 hash，`-1` 表示尚无稳定内容（partial 块） |
| `token_ids` | 该块实际存的 token 序列，用于 hash 碰撞时二次校验 |

`reset()` 把 `ref_count` 置为 **1** 而不是 0——因为 reset 总是在"新分配给某条 seq"的那一刻调用，第一个使用者就位。

### 1.2 BlockManager 的四个池

| 结构 | 类型 | 存什么 | 不变量 |
|------|------|--------|--------|
| `blocks` | list | 全部 `num_blocks` 个 `Block` | 下标 == `block_id`，长度永不变 |
| `free_block_ids` | deque | 空闲 block_id | 和 used 互补且不相交 |
| `used_block_ids` | set | 在用 block_id | 同上 |
| `hash_to_block_id` | dict | hash → block_id | hash 碰撞靠 `token_ids` 二次校验兜底 |

**核心不变量**：`free_block_ids ∪ used_block_ids == {0..num_blocks-1}` 且 `∩ == ∅`。

**为什么选这些容器？**
- free 要"取一个/还一个"，deque 两端 O(1)。
- used 要 `in` 查询，set 全 O(1)。
- hash_to_block_id 是全局去重表，**一旦写入永不删除**（deallocate 不清它）。

---

## 2. 级联哈希 `compute_hash`

```python
h = hash(prefix_hash || token_ids)
```

把前一块的 hash 拌进这一块的 hash 计算。**"第 k 块的 hash"=整段 [0..k] 的内容摘要**。

**为什么级联？** 如果每块单独 hash，不同 seq 的"第 2 块"内容相同就会被当成同一个物理块——前面 1~4 块完全不同也被硬复用，attention 读到错误历史。级联之后，只有**从开头一路相同**的两条 seq 才会在第 k 块命中。这才是 prefix cache 的正确语义。

配合 `hash_to_block_id.get(h) → block_id`，再用 `self.blocks[block_id].token_ids != token_ids` 做二次校验（防 xxhash 碰撞）。

---

## 3. 两个私有原子操作

```python
_allocate_block(block_id):
    assert ref_count == 0
    block.reset()                  # ref_count=1, hash=-1, token_ids=[]
    free → used

_deallocate_block(block_id):
    assert ref_count == 0
    used → free
    # 注意：不清 block.hash 和 block.token_ids！
```

所有上层 API 最终都落到这两个函数上。

**关键细节**：`_deallocate_block` 只搬运 block_id 在 free/used 两池间，**完全不碰 block 元数据**。这就是"僵尸块"存在的根本原因——见 §7。

---

## 4. Prefill 时的分配：`can_allocate` + `allocate`

### 4.1 `can_allocate` 是保守策略

```python
return len(self.free_block_ids) >= seq.num_blocks
```

**不看 prefix cache**。即使 prompt 和历史 100% 命中（理论 0 个新块），也按最坏情况"全部 miss"判断。代价是偶尔误拒，收益是避免分配中途失败回滚。

### 4.2 `allocate` 的三条分支

对每个 i-th 块，按顺序判断：

| 分支 | 条件 | 动作 |
|------|------|------|
| **cache_miss** | hash 查不到 或 `blocks[bid].token_ids != token_ids` 或前一块已 miss | `_allocate_block(free[0])`，ref=1 |
| **hit + in used** | hash 命中 且 `bid in used_block_ids` | `ref_count += 1`；不动 free/used 池 |
| **hit + 不在 used**（"僵尸块复活"） | hash 命中 但 `bid in free_block_ids` | `_allocate_block(bid)` 从 free 拿回来，复用原内容 |

后两种都是"prefix cache 命中"：`seq.num_cached_tokens += block_size`，这一块的 K/V 不用重算。

### 4.3 cache_miss "传染"

```python
if cache_miss:
    ...
```

一旦某块 miss，后续所有块必然 miss（级联 hash 里混了前面的"新 hash"，查不到）。这是正确行为，不是妥协。

---

## 5. Decode 时的增长：`can_append` + `may_append`

### 5.1 `can_append` 的巧妙公式

```python
return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)
```

布尔值 True=1, False=0，正好表达"需要几个新块"：

- `len % 256 == 1`：刚跨过块边界（第 257 个 token），需要 1 个新块。
- 其他情况：在当前块内续写，不需要新块。

### 5.2 `may_append` 的三条分支

| `len(seq) % 256` | 含义 | 动作 |
|-----------------|------|------|
| **== 1** | 刚开始写新块 | `assert 上一块 hash != -1`；从 free 拿新块 append 到 block_table |
| **== 0** | 最后一块刚好填满 | `assert 它 hash == -1`；计算 hash，`update + 登记 hash_to_block_id` |
| **其他** | 部分块继续填 | `assert hash == -1`；无物理操作 |

### 5.3 为什么 hash 要等"刚填满"那一刻才算？

Decode 是**边生成边填**。只有最后一个槽写完、这块的全部 256 个 token 确定下来，算出来的 hash 才有意义。部分块内容还在变化，不参与 prefix cache。

对比 `allocate`：prefill 时 seq 的全部 token 已知，可以一次性对所有满块算 hash。

---

## 6. 释放：`deallocate`

```python
for block_id in reversed(seq.block_table):
    block.ref_count -= 1
    if block.ref_count == 0:
        _deallocate_block(block_id)
seq.num_cached_tokens = 0
seq.block_table.clear()
```

几个关键点：

- **`reversed` 的意义**：序列末端的块多为独占（`ref=1`，最近刚分的），最可能真正归还；前面的 prefix 块更多是共享（`ref>1`）。逆序先处理容易归还的，free 池快速变大，对并发友好。在当前同步调度里属于直觉优化。
- **`ref_count` 降到 0 才真正归还**：保证共享块只要还有人用就不会被回收。
- **`num_cached_tokens=0` + `block_table=[]`**：对应抢占场景——下一次 `schedule()` 时进 prefill 分支重建。
- **不清 hash 和 token_ids**：这是僵尸块的关键。

---

## 7. 僵尸块（Resurrection Mechanism）

**定义**：一个块在 free 池里，但它自己还记得"我上次装的是哪段 token、hash 是多少"，`hash_to_block_id` 也还记得"这段 hash 指向我"。

**时间线**：

```
T1  seq A allocate → block 0 装 [A,B,C,D], hash=h0, ref=1, in used, hash_to_block_id[h0]=0
T2  seq A finish   → deallocate: ref 1→0, _deallocate_block(0)
                     block 0 → free; 但 block.hash=h0, token_ids=[A,B,C,D] 仍在
                     hash_to_block_id[h0]=0 仍在
T3  seq B allocate 同样的 prompt 头
    compute_hash → h0, 查表命中 0
    blocks[0].token_ids == [A,B,C,D] ✓ 二次校验过
    0 in used_block_ids? 否（在 free）
    → 走"复活"分支：_allocate_block(0) 把它从 free 拿回来
      先 reset() 清空，立刻 update(h0, [A,B,C,D]) 恢复内容
    → B.num_cached_tokens += 4
```

两种命中分支对比：

| 分支 | 条件 | 效果 |
|------|------|------|
| `in used_block_ids → ref_count += 1` | 有其他 seq 还在用 | 白嫖引用，不动池子 |
| 不在 used → `_allocate_block` | 上一用户已释放，但内容还在 | 从 free 复活 |

**副作用**：`hash_to_block_id` 会无限增长（含大量指向"已被别人重新分配、内容已覆盖"的旧条目）。靠 `allocate` 里的 `self.blocks[block_id].token_ids != token_ids` 二次校验兜住——hash 命中但内容对不上当 miss。生产级 vLLM 会加 LRU 清理；nano-vllm 为了简洁没做。

---

## 8. ref_count 的完整生命周期

| 场景 | ref_count 变化 | 触发函数 |
|------|-------------|---------|
| 新块首次分配 | `reset()` 置 1 | `_allocate_block` |
| Prefix cache 命中（共享） | +1 | `allocate` 的 `in used` 分支 |
| 僵尸块复活 | `reset()` 置 1 | `_allocate_block` |
| seq 结束/被抢占 | -1 | `deallocate` |
| 降到 0 | 块真正归还 | `deallocate` 中触发 `_deallocate_block` |

---

## 9. 端到端例子：3 条 seq 的完整生命周期

设定：`block_size=4`（方便心算），`num_blocks=8`，`max_num_seqs=3`，`max_num_batched_tokens=32`。

| seq | prompt | 生成几 token | 特征 |
|-----|--------|------------|------|
| A | `[A,B,C,D, E,F,G]`（7） | 5 | — |
| B | `[A,B,C,D, X,Y]`（6） | 3 | **共享 A 的前 4 token** |
| C | `[K,L,M,N, O]`（5） | 4 | 独立 |

### T1 prefill 阶段

allocate 三条 seq：

**A**：cache_miss 所有块
- block 0 ← [A,B,C,D]，hash=h0，登记
- block 1 ← [E,F,G] partial，不登记
- `A.block_table=[0,1]`，`num_cached=0`

**B**：prefix cache 命中
- i=0: [A,B,C,D] → hash=h0 → 命中 block 0 ∈ used → **ref_count: 1→2**，`B.num_cached=4`
- i=1: [X,Y] partial → miss → block 2，ref=1
- `B.block_table=[0,2]`，`num_cached=4`，`num_scheduled=2`

**C**：全部 miss
- block 3 ← [K,L,M,N]，登记 h_c0
- block 4 ← [O] partial
- `C.block_table=[3,4]`

T1 状态快照：
```
used={0,1,2,3,4}, free={5,6,7}
ref_count: [0:2, 1:1, 2:1, 3:1, 4:1]
hash_to_block_id={h0:0, h_c0:3}
```

前向写入 slot_mapping 对应的 K/V，Sampler 产出 `[a1, b1, c1]`。postprocess append，A/B/C 长度变为 8/7/6。

### T2 decode 1

- A len=8，`%4==0` → `may_append` 计算 block 1 的 h_A1（prefix=h0），登记
- B len=7，`%4==3` → else，无操作
- C len=6，`%4==2` → else，无操作

前向 store_kvcache 写 a1/b1/c1 的 K/V，采样产 `[a2, b2, c2]`。append 后长度 9/8/7。

### T3 decode 2 —— 新块 + B 结束

- A len=9，`%4==1` → **从 free 取 block 5** 加入 `A.block_table=[0,1,5]`
- B len=8，`%4==0` → 算 h_B1 登记 block 2
- C len=7，`%4==3` → else

前向 + 采样。B 生成第 3 个 completion 达到 max，**finish**。

`deallocate(B)`，逆序 `[2, 0]`：

| block_id | ref 前 | ref 后 | 归还？ |
|----------|-------|-------|-------|
| 2 | 1 | 0 | **是** |
| 0 | 2 | 1 | 否（A 还在用） |

T3 末状态：`used={0,1,3,4,5}, free={2,6,7}`。`hash_to_block_id` 不动，block 2 成为僵尸块，`blocks[2].token_ids=[X,Y,b1,b2]` 保留。

### T4 decode 3 —— C 结束

C 达到 max。`deallocate(C)` 释放 block 3、4 → free。
`used={0,1,5}`，free={2,3,4,6,7}。

### T5 A 收官

A 再 decode 几步到 max。`deallocate(A)` 释放 block 0、1、5。
最终 `used={}`，`free={0..7}`。

`hash_to_block_id` 保留 `{h0, h_A1, h_B1, h_c0, h_C1}` 共 5 条僵尸条目，等着后续同前缀的 seq 命中复活。

---

## 10. 进阶场景

### 10.1 Chunked prefill 过程中被 preempt 会怎样？

**结论**：正在 chunked prefill 的那条 seq 本身不会被 preempt。

原因：
1. `preempt` 只在 decode 分支调用，且只动 `running` 队列。
2. chunked 中的 seq 状态是 WAITING，留在 `waiting` 头部，不在 preempt 视野内。
3. `schedule()` 一旦 prefill 分支有任务就 return，decode 根本不跑。

**但有间接影响**：别的 running seq 被 preempt 后，`waiting.appendleft` 把它塞到 waiting 最前面，**插到 chunked 老任务前面**。被 preempt 的 seq 自己也常常需要重走 chunked prefill——因为它的 `token_ids` 已包含 prompt + 大量 completion，长度可能超预算。这时 prefix cache 帮大忙：prompt 部分的级联 hash 还在（块多半是僵尸），`allocate` 里直接命中，`num_cached_tokens` 在那一步就累加到接近 prompt 长度，真正要重算的只是已生成的 completion tokens。

**注意 chunked 中 seq 的块是"冻结"的**：它既不在 running、也没 finish，没有任何代码路径会释放它的块。这是 chunked prefill 的显式代价。

### 10.2 多条 seq 同时 `%==1`、free 不够：谁先谁后？

**结论**：running FIFO 顺序，头部优先；尾部被砍。

代码依据：
```python
seq = self.running.popleft()          # 从头取
while not can_append(seq):
    if self.running:
        self.preempt(self.running.pop())   # 从尾砍
```

走一遍 running=[A,B,C,D]、全部 `%4==1`、free=0：

| 步骤 | 当前 seq | 动作 |
|-----|---------|------|
| 1 | A popleft | can_append False → preempt D（tail）→ D 的块全归还 → can_append True → may_append(A) |
| 2 | B popleft | free 已充裕 → may_append(B) |
| 3 | C popleft | may_append(C) |
| 4 | running 空 | 退出 |

最终：`scheduled_seqs=[A,B,C]`，D 被扔回 `waiting.appendleft`。

**只抢一条就够**——D 释放的多块够后面 B、C 消费。preempt 是粗粒度释放。

**两条原则**：
- FIFO 公平：老请求先 decode 先 finish。
- 抢占局部性：running 尾部是刚进来的，completion 少，重 prefill 代价低（prompt 部分 prefix cache 覆盖）。

---

## 11. block_size 的折衷

| 维度 | block_size=256 | block_size=1 |
|------|---------------|--------------|
| per-token allocate 次数 | 每 256 token 1 次 | 每 token 1 次 |
| may_append 算 hash | 少 | 每 token 1 次 |
| `block_table` 长度 | seq_len / 256 | seq_len（爆炸） |
| GPU 上 slot_mapping / block_tables 张量 | 小 | 巨大 |
| `hash_to_block_id` 条目数 | 少 | 数十倍膨胀 |
| prefix cache 粒度 | 256 tokens | 1 token（理论更细） |
| GPU 指针追踪（paged attn） | 少 | 每算一个 token 查一次 block_table |

block_size 太大（4096）也不好：10 token 的短 prompt 也占 4096 槽，浪费显存。**256 是 prefix cache 粒度、kernel 效率、显存利用率之间的折衷**。

---

## 12. 关键不变量速查

1. `free_block_ids ∪ used_block_ids == {0..num_blocks-1}`，交集为空。
2. `used` 里的 block：`ref_count ≥ 1`；`free` 里的 block：`ref_count == 0`。
3. `hash_to_block_id` 只增不减（写入后不删除）；依赖 `allocate` 里 `token_ids` 二次校验容错。
4. `block.hash != -1` 表示该块内容已满且稳定（可被 prefix cache 复用）；`-1` 表示部分块。
5. `seq.block_table[i]` 对应 seq 第 i 个逻辑块（每块 `block_size` tokens）。
6. `seq.num_cached_tokens` 仅记录 prefix cache **命中的整块数 × block_size**（partial 块即便部分命中也不计）。
7. `_allocate_block` 和 `_deallocate_block` 严格匹配：前者 free→used 且 ref 置 1；后者 used→free 且必须 ref 已降到 0。

---

## 13. 两条脉络的合流图

```
调度层（Day 1）                    显存层（Day 2）
─────────────                     ─────────────
schedule prefill 选 seq
       │
       ├─ 首次 prefill ──────────▶ allocate(seq)
       │                              ├─ 遍历 num_blocks 个块
       │                              ├─ 级联 hash 查表
       │                              ├─ miss → _allocate_block
       │                              ├─ hit+used → ref_count++
       │                              ├─ hit+free → 复活（_allocate_block）
       │                              └─ 写入 block_table / num_cached_tokens

prepare_prefill
  └─ 用 block_table 算 slot_mapping ◀── 消费
  
前向 store_kvcache → 写入物理槽位

schedule decode 对每条 seq
       │
       ├─ can_append 检查空间
       │   └─ 不够 → preempt running.pop ────▶ deallocate(被抢占 seq)
       │                                           ├─ 逆序 ref_count - 1
       │                                           └─ 0 时 _deallocate_block
       └─ may_append 按三分支
             ├─ %==1 → 从 free 取新块
             ├─ %==0 → 算 hash 登记
             └─ else → 无操作

postprocess finish
  └─ deallocate(seq) ← 同抢占路径，差别仅在不 appendleft 回 waiting
```

---

## 14. 本日压缩成 5 句话

1. **BlockManager 是块池的管家**，四个数据结构 `blocks / free / used / hash_to_block_id` 的不变量决定一切行为。
2. **级联 hash + 二次校验**是 prefix cache 正确性的两道保险，hash 碰撞和僵尸块歧义都在这两步兜住。
3. **`ref_count` 控生命**：共享时 +1，释放时 -1，降到 0 才真归还；`deallocate` 的 `reversed` 是为了让独占块先归还。
4. **僵尸块机制**：`_deallocate_block` 不清元数据，让已释放的块仍能被同前缀的新 seq 命中复活——prefix cache 跨 seq 延续的关键。
5. **调度几何**：waiting 左进右出、running 左进左出-右抢，preempt 把 running 尾扔回 waiting 头；chunked prefill 中的 seq 本身免疫 preempt，但会被新 preempt 进来的 seq 插队。
