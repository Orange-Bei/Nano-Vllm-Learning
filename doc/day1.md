# Day 1 学习笔记：Nano-vLLM 一次推理的完整数据流

本笔记以 `example.py` 为起点，串起一次 `llm.generate()` 从 prompt 到 output 的全过程，涵盖调度、KV 管理、ModelRunner、Attention、采样、抢占与 chunked prefill。目标是建立一张可以作为后续几天导航的全景图。

---

## 0. 运行场景

`example.py:24` 的调用如下：

```python
llm = LLM(path, enforce_eager=True, tensor_parallel_size=1)
sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
prompts = ["introduce yourself", "list all prime numbers within 100"]
outputs = llm.generate(prompts, sampling_params)
```

两条 prompt 经 chat template 处理、tokenize 后，进入 `LLMEngine.generate()`。

---

## 1. 顶层驱动：`LLMEngine.generate()`

`nanovllm/engine/llm_engine.py:60`

1. 对每条 prompt 调 `add_request()`：`Sequence(prompt, sampling_params)` 得到 `seq0`、`seq1`，状态 `WAITING`，进入 `scheduler.waiting`。
2. 进入 `while not is_finished(): step()` 主循环，用 tqdm 显示 prefill / decode 吞吐。
3. 每次 `step()` 返回已完成的 seq 的 token 列表，最后 tokenizer.decode 得到文本。

`step()` 的骨架 (`llm_engine.py:49`)：

```python
seqs, is_prefill = self.scheduler.schedule()
token_ids = self.model_runner.call("run", seqs, is_prefill)
self.scheduler.postprocess(seqs, token_ids, is_prefill)
```

主线路：**Scheduler 决定跑谁 → ModelRunner 准备张量并前向 → Sampler 采样 → Scheduler 回写状态**。

---

## 2. 调度层：`Scheduler.schedule()`

`nanovllm/engine/scheduler.py:27`

### 2.1 Prefill 优先

从 `waiting` 队列取 seq：

- `num_tokens_needed = max(seq.num_tokens - seq.num_cached_tokens, 1)`：还要跑多少 token。
- `remaining = max_num_batched_tokens - num_batched_tokens`：本步 token 预算。
- `block_table` 为空时调用 `block_manager.allocate(seq)` 分配 KV 块。
- `seq.num_scheduled_tokens = min(num_tokens_needed, remaining)`：本步实际要跑的片段。
- 若 `num_scheduled_tokens == num_tokens_needed` → 状态转 `RUNNING`，挪到 `running` 队列；否则仍留在 `waiting`（chunked prefill 中）。
- 一旦本步有 prefill，直接 `return (seqs, True)`，**prefill 与 decode 不在同一步混跑**。

### 2.2 Decode

`waiting` 为空或已没预算时进入 decode 分支：

- 从 `running` 取所有在跑序列，每条调度 1 个 token。
- `block_manager.can_append(seq)`：若需要新块但池里没有，调用 `preempt()` 把序列踢回 `waiting`（释放其块）。

### 2.3 Postprocess

`scheduler.py:81`

- Prefill 阶段：`num_cached_tokens += num_scheduled_tokens`；若未完（chunked 没跑完）或已有 completion token（抢占后重 prefill），则 `num_scheduled_tokens=0` 然后 `continue`，**不 append_token**。
- Decode 阶段：`seq.append_token(token_id)`；判 EOS / `max_tokens` → 转 `FINISHED`，归还 KV 块。

---

## 3. 显存层：`BlockManager`

`nanovllm/engine/block_manager.py`

- 整个 KV cache 被切成 `num_blocks` 个固定大小的 `Block`，默认 `block_size=256`。
- `Sequence.block_table` 是一个 `list[int]`，把 seq 的逻辑位置映射到物理 block id。
- **Prefix cache** 靠 `xxhash` + 级联前缀 hash 实现：每个满块算一个 hash，`hash_to_block_id` 命中时 `ref_count += 1` 并 `num_cached_tokens += block_size`，复用历史块的 K/V。
- 部分块（不满 256 tokens）不参与 prefix cache，`hash = -1`。
- `deallocate()`：倒序递减 `ref_count`，降到 0 才真正归还；同时清零 `seq.num_cached_tokens` 和 `seq.block_table`。

---

## 4. 执行层：`ModelRunner`

`nanovllm/engine/model_runner.py`

### 4.1 Prefill 输入准备 `prepare_prefill` (`model_runner.py:129`)

把所有 seq 的待算 token 拼成 1-D 大张量（无 batch 维）：

- `input_ids`：`token_ids[start:end]` 拼接。
- `positions`：每条 seq 从 `num_cached_tokens` 处开始。
- `cu_seqlens_q`：本步每条 seq 的 Q 长度累积偏移，给 flash_attn_varlen。
- `cu_seqlens_k`：每条 seq 的 K 长度累积偏移（完整历史长度）。
- `slot_mapping`：每个新 token 在物理 KV 池的槽位 = `block_id * 256 + 块内偏移`。
- `block_tables`：若有 prefix 或 chunked prefill（`cu_seqlens_k > cu_seqlens_q`），补齐到相同列数。

### 4.2 Decode 输入准备 `prepare_decode` (`model_runner.py:173`)

每条 seq 只取 `last_token`：

- `input_ids`：`[seq.last_token for seq in seqs]`。
- `positions`：`len(seq) - 1`。
- `slot_mapping`：`block_table[-1] * 256 + last_block_num_tokens - 1`。
- `context_lens`：各序列完整长度。
- `block_tables`：已填好的块索引。

### 4.3 全局 Context

`nanovllm/utils/context.py`

准备好的张量通过 `set_context(...)` 写入全局 `_CONTEXT`，Attention 层 `get_context()` 取用。每步结束 `reset_context()` 清空。

### 4.4 CUDA Graph

`enforce_eager=False` 时 decode 走 `capture_cudagraph()` 预录好的固定 bs graph（1,2,4,8,16,32,…），`graph.replay()` 省 launch 开销。prefill 因 token 数变化大不走 graph。

---

## 5. 模型前向：`Qwen3ForCausalLM`

`nanovllm/models/qwen3.py`

```
embed_tokens(input_ids)             # [N, hidden]
for layer in num_hidden_layers:
    input_layernorm
    self_attn (Qwen3Attention)
    post_attention_layernorm
    mlp (SiluAndMul + gate_up / down_proj)
norm
compute_logits → lm_head            # [N, vocab]
```

`Qwen3Attention.forward` (`qwen3.py:72`)：

```python
qkv = qkv_proj(hidden)
q, k, v = split → reshape [N, H, D]
q = q_norm(q); k = k_norm(k)        # Qwen3 独有的 q/k RMSNorm
q, k = rotary_emb(positions, q, k)  # RoPE
o = self.attn(q, k, v)              # → layers/attention.py
o_proj(o.flatten(1, -1))
```

---

## 6. Attention：paged KV 的核心

`nanovllm/layers/attention.py`

两步：

### 6.1 写 KV cache

```python
store_kvcache(k, v, self.k_cache, self.v_cache, context.slot_mapping)
```

Triton kernel 每个 program 处理一个 token，按 `slot_mapping[idx]` 把 k/v 直接 `tl.store` 到物理位置。`self.k_cache` 是 `allocate_kv_cache` 预分配的 view，形状 `[num_blocks, 256, num_kv_heads, head_dim]`，每层一份。

### 6.2 算 attention

- **Prefill**：`flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, causal=True, block_table=...)`。无 prefix 时 k/v 直接用刚写入的那一批；有 prefix 或 chunked 时 k/v 换成 `k_cache, v_cache`，靠 `block_table` 读历史。
- **Decode**：`flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache, cache_seqlens=context_lens, block_table=..., causal=True)`。Q shape `[bs, 1, H, D]`，flash-attn 按块读整段历史。

---

## 7. 采样：`Sampler`

`nanovllm/layers/sampler.py`

```python
logits = logits.float().div_(temperatures.unsqueeze(1))
probs  = softmax(logits, dim=-1)
tokens = probs.div_(Exp(1) 噪声).argmax(-1)   # Gumbel-max 等价于按温度采样
```

返回 `[num_seqs]` 的 token_id 列表。

---

## 8. 抢占与重 prefill

### 8.1 触发

decode 阶段 `can_append(seq)` 判断需要新块但 `free_block_ids` 为空时，`preempt()` 把另一条 running seq（或自己）退回 waiting。

### 8.2 被抢占 seq 的状态

```
preempt → block_manager.deallocate(seq):
    seq.num_cached_tokens = 0
    seq.block_table.clear()
status = WAITING, appendleft to waiting
```

`token_ids` 保留（prompt + 已生成 completion）。

### 8.3 重新调度

- `block_table` 空 → 重新 `allocate`，prefix cache 命中则复用历史块。
- `num_tokens_needed = len(token_ids)`：对整段（prompt + completion）重做 prefill。
- 重 prefill 完成后，`postprocess` 里 `num_completion_tokens > 0` 分支触发 `continue`，**不 append 新 token**，只是把 KV 重建出来。
- 下一步才继续 decode。

---

## 9. Chunked prefill 是真实现了的

关键证据散落在三个文件：

- **scheduler.py:37-52**：
  - `num_scheduled_tokens = min(num_tokens, remaining)` 才是真正的切片动作。
  - 只有 `num_scheduled_tokens == num_tokens` 时才转 `RUNNING`、`popleft`；否则 seq 仍在 waiting 头部，下一轮继续。
  - 注释 `only allow chunked prefill for the first seq`：预算不够时只给队首切片，后续 seq 等下轮。
- **scheduler.py:83-87**：未跑完的 chunked step 不 append_token。
- **model_runner.py:140-147**：`start = min(num_cached_tokens, seqlen-1)` 从上次停的地方继续；`seqlen_k = seqlen` 让 flash-attn 读完整历史。
- git log `8d63a98 support chunked prefill` 是加这套逻辑的 commit。

---

## 10. 完整 Walkthrough：30000-token prompt, max_num_batched_tokens=8192

### 10.1 前置

- `block_size=256`，需要 `⌈30000/256⌉ = 118` 个块（117 满块 + 1 个装 48 tokens）。
- 预算切分：`8192 + 8192 + 8192 + 5424 = 30000`，共 4 步 prefill + 后续 decode。

### 10.2 初始状态

```
seq.token_ids         = [t0, t1, ..., t29999]
seq.num_tokens        = 30000
seq.num_cached_tokens = 0
seq.num_scheduled_tokens = 0
seq.block_table       = []
seq.status            = WAITING
waiting = [seq], running = []
```

### 10.3 Step 1

**Scheduler**：
```
num_tokens_needed = 30000, remaining = 8192
allocate(seq): 118 块全 cache_miss → block_table = [B0..B117], num_cached_tokens 仍 0
num_scheduled_tokens = min(30000, 8192) = 8192
8192 != 30000 → 保持 WAITING
```

**prepare_prefill**：
```
start=0, end=8192, seqlen_q=8192, seqlen_k=30000
input_ids  = token_ids[0:8192]
positions  = [0..8191]
cu_seqlens_q = [0, 8192]
cu_seqlens_k = [0, 30000]
slot_mapping 覆盖 B0..B31 的 8192 个槽
block_tables = [[B0..B117]]
```

**前向**：Triton 写 B0..B31 的 K/V；flash_attn_varlen_func 用 block_table 读历史。

**postprocess**：`num_cached_tokens = 8192`，`num_scheduled_tokens = 0`，continue（不 append）。

### 10.4 Step 2

```
num_tokens_needed = 30000 - 8192 = 21808, remaining = 8192
num_scheduled_tokens = 8192, 仍 WAITING
start=8192, end=16384, positions=[8192..16383]
slot_mapping 覆盖 B32..B63
postprocess: num_cached_tokens = 16384
```

### 10.5 Step 3

```
num_tokens_needed = 13616, num_scheduled_tokens = 8192
start=16384, end=24576, positions=[16384..24575]
slot_mapping 覆盖 B64..B95
postprocess: num_cached_tokens = 24576
```

### 10.6 Step 4（最后一片）

**Scheduler**：
```
num_tokens_needed = 5424, remaining = 8192
num_scheduled_tokens = 5424
5424 == num_tokens_needed → status = RUNNING, waiting.popleft(), running.append(seq)
```

**prepare_prefill**：
```
start=24576, end=30000, seqlen_q=5424
positions=[24576..29999]
slot_mapping:
  i=96..116  : 21 个整块 × 256 = 5376 slot
  i=117(last): B117*256 + 0 .. B117*256 + 48 共 48 slot
  total = 5424
```

**postprocess**（走正常分支）：
```
num_cached_tokens = 30000
num_completion_tokens == 0 且 num_cached_tokens == num_tokens → 不 continue
append_token(t_next)     # token_ids 变 30001
num_cached_tokens += 1   # 30001
status 保持 RUNNING
```

seq 正式进入 decode 阶段，下一步走 `prepare_decode` + `flash_attn_with_kvcache`。

### 10.7 变量速查表

| step | start | end | seqlen_q | num_cached_tokens（后） | slot 覆盖的块 | status（后） | 队列 |
|------|-------|-----|----------|----------------------|--------------|--------------|------|
| 1 | 0 | 8192 | 8192 | 8192 | B0..B31 | WAITING | waiting |
| 2 | 8192 | 16384 | 8192 | 16384 | B32..B63 | WAITING | waiting |
| 3 | 16384 | 24576 | 8192 | 24576 | B64..B95 | WAITING | waiting |
| 4 | 24576 | 30000 | 5424 | 30001 | B96..B117 | RUNNING | running |

### 10.8 关键不变量

1. **`cu_seqlens_q` 每步重算**，Q 长度按本片；`cu_seqlens_k` 永远是 `len(seq)`，flash-attn 靠 `block_table` 读完整历史。
2. **`num_scheduled_tokens`** 只在 schedule 时写、postprocess 时清零，是"本步跑多少"的临时游标。
3. **`num_cached_tokens`** 单调递增到等于 `num_tokens`，是"KV 里已有多少真实数据"的游标；`prepare_prefill` 靠它定位下一片起点。
4. **`block_table` 在第 1 步 allocate 时就全部分好**，后续 3 步只在未写入的块上继续填 K/V，不再 allocate。

---

## 11. 一张图总结

```
prompt str
  └─tokenize→ token_ids ──add_request──▶ Sequence(WAITING) → scheduler.waiting
                                                    │
                                    ┌── scheduler ──┘
                                    ▼
                (prefill) allocate KV blocks (若 block_table 空)
                                    │
                                    ▼
                    prepare_prefill / prepare_decode
                    → Context(slot_mapping, cu_seqlens, block_tables, ...)
                                    │
                                    ▼
                    Qwen3ForCausalLM 前向，每层 Attention：
                        ├─ store_kvcache (Triton) → paged KV
                        └─ flash_attn_varlen / with_kvcache
                                    │
                                    ▼
                    compute_logits → Sampler → next_token_id
                                    │
                                    ▼
                    postprocess：
                      - prefill 未完 / 抢占重 prefill → 不 append
                      - 正常 decode → append_token / 判 EOS / 释放块
                                    │
                             未结束 ─▶ 下一步
                             结束   ─▶ tokenizer.decode → output text
```

---

## 12. 六视角回顾

- **请求层**：`Sequence` / `SamplingParams` 承载 prompt、采样配置、运行期状态。
- **调度层**：`Scheduler` 决定 prefill / decode / 抢占，维护 `waiting` 与 `running`。
- **显存层**：`BlockManager` 以 256 token 为单位管 KV cache 和 prefix cache。
- **执行层**：`ModelRunner` 把逻辑请求转成张量，写全局 `Context`，执行 eager 或 CUDA Graph。
- **并行层**：线性层列/行并行、词表并行，rank 0 控制面、其他 rank 走 `loop()` 执行面（本例 TP=1 未涉及）。
- **模型层**：`Qwen3ForCausalLM` + `layers/` 下的 attention/RMSNorm/RoPE/SiluAndMul。

框架真正的价值集中在前四层：**一样的模型，换上调度 + paged KV + continuous batching + CUDA Graph，吞吐就是几倍的差距**。
