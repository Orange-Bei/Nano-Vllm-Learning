# 项目总回顾：一次真实推理的完整生命周期

> 用一个具体场景把 Day 1–Day 5 的所有知识点串成一条线：从进程启动，到请求调度，到 TP 多卡前向，到 prefix cache 命中，到 CUDA Graph decode，再到退出清理。

---

## 0. 场景设定

```python
# example_full.py
from nanovllm import LLM, SamplingParams

llm = LLM("Qwen/Qwen3-4B", tensor_parallel_size=2, enforce_eager=False)

prompts = [
    "你是一个AI助手。" * 40 + "请介绍 PagedAttention 的工作原理",   # A: 共 340 tokens
    "你是一个AI助手。" * 40 + "请介绍 FlashAttention 的工作原理",    # B: 共 340 tokens
]
sampling_params = [SamplingParams(temperature=0.6, max_tokens=50)] * 2
outputs = llm.generate(prompts, sampling_params)
```

**关键数字**（贯穿全程用）：
- 2 张 GPU，`TP=2`，`block_size=256`，`max_num_seqs=512`
- A 和 B 的前 320 tokens **完全相同**（系统提示）→ 一个完整 block 可命中 prefix cache
- A、B 的 tokens 256-319 是共享系统提示尾巴，320-339 是各自不同的用户问题

---

## 1. 冷启动阶段（Day 5）

```
主进程（rank 0）                              子进程（rank 1，spawn 出来）
────────────────                              ──────────────────────
LLMEngine.__init__
  mp.spawn(ModelRunner, rank=1) ───────────►  ModelRunner(config, rank=1, event)
  ModelRunner(config, rank=0, events)               │
     │                                              │
     ▼                                              ▼
  ┌─────────────────── 两个 rank 对称做 6 件事 ───────────────────┐
  │ 1. dist.init_process_group("nccl", tcp://localhost:2333)     │
  │ 2. Qwen3ForCausalLM(hf_config) — TP 切片的模型骨架           │  ← Day 3-4
  │ 3. load_model — 从 safetensors 读权重，按 weight_loader 切片  │  ← Day 3-4
  │ 4. warmup_model — 最大 batch 假 prefill，探显存峰值           │  ← Day 5
  │ 5. allocate_kv_cache — 显存公式算出 num_kvcache_blocks        │  ← Day 5
  │ 6. capture_cudagraph — 36 个 bs 挡位预录 decode graph         │  ← Day 5
  └──────────────────────────────────────────────────────────────┘
     │                                              │
     ▼                                              ▼
  shm = SharedMemory(create=True, 1MB)          shm = SharedMemory(open)
  tokenizer, scheduler, atexit.register              │
                                                     ▼
                                                  loop() — 阻塞 event.wait()
```

**这时两张卡的状态**：
- **模型权重**：每 rank 持一半 attention heads + 一半 MLP 中间维 + 一半 vocab
- **KV cache**：每张卡各自分了比如 500 个 block（因为 `num_kv_heads` 也被切一半，每块更小）
- **CUDA Graph**：36 份 graph 对象等着被 replay
- **SharedMemory**：rank 0 写 rank 1 读的 1MB 控制通道

---

## 2. 接收请求（Day 1）

```python
llm.generate(prompts, sampling_params)
  └─ for prompt in prompts: self.add_request(prompt, sp)
         ├─ token_ids = tokenizer.encode(prompt)   # [128001, 48, 35783, ..., 42069]  340 个
         ├─ seq = Sequence(token_ids, sp)          # seq.status = WAITING
         └─ scheduler.add(seq)                     # 进入 waiting 队列
```

此时 scheduler：`waiting=[A, B]`，`running=[]`。

---

## 3. 第一个 step：Prefill 阶段（最精彩的一步）

```python
while not llm.is_finished():
    output = llm.step()
```

### 3.1 Scheduler 决策（Day 2：BlockManager）

```
Scheduler.schedule()
  └─ 看 waiting 队列非空 → 走 prefill 分支
     └─ 对每个 seq 调 block_manager.can_allocate / allocate
```

**处理 A（先来先处理）**：

```
block_manager.allocate(A):
  A.token_ids 切块：
    block_0 = tokens[0:256]   ← 完整块
    block_1 = tokens[256:340] ← 部分块（84 tokens）

  hash_0 = xxhash(prefix=-1, tokens[0:256])
  查 hash_to_block_id:
    MISS → 从 free_block_ids 弹出一个 block_id=5，block_table[0]=5
    hash_to_block_id[hash_0] = 5

  block_1 是部分块，不 hash，分配 block_id=6
  block_table = [5, 6]
  num_cached_tokens = 0    # A 是第一个请求，无缓存可用
```

**处理 B（重点看 prefix cache）**：

```
block_manager.allocate(B):
  hash_0 = xxhash(prefix=-1, tokens[0:256])
  查 hash_to_block_id:
    HIT → block_id=5（就是 A 刚才占的那块！）
    increment ref_count[5]   ← 两个 seq 共享块 5
    block_table[0] = 5
    ✨ num_cached_tokens = 256  ← 前 256 个 token 不用重算

  block_1 部分块，分配 block_id=7
  block_table = [5, 7]
```

**调度结果**：

```
seq A: num_cached_tokens=0,   num_scheduled_tokens=340
seq B: num_cached_tokens=256, num_scheduled_tokens=340-256=84   ← 只算后 84 个！
总 token 预算：340+84=424 < max_num_batched_tokens=16384 ✓
```

### 3.2 跨进程 RPC（Day 5）

```
rank 0: model_runner.call("run", [A, B], is_prefill=True)
    ├─ write_shm: pickle.dumps(["run", [A, B], True]) → 写入共享内存
    │              （Sequence 的 __getstate__ 只 pickle 必要字段，省带宽）
    ├─ for event in events: event.set()   ──control──►  rank 1 unblocks
    └─ 自己也调 self.run(...)                                       │
                                                                    ▼
                                         rank 1: read_shm 读出方法名和参数
                                         rank 1: self.run([A, B], True)
```

两个 rank 从这里开始**对称**执行，只在通信层交汇。

### 3.3 张量准备（Day 1 + Day 2）

每个 rank 都跑一遍 `prepare_prefill`：

```python
# 只有 B 需要跳过前 256 tokens（已缓存），A 从头算
input_ids   = [A.tokens[0:340], B.tokens[256:340]]  # flat 拼成 [424]
positions   = [0..339, 256..339]                    # [424]
cu_seqlens_q = [0, 340, 424]                        # 每条序列的边界
cu_seqlens_k = [0, 340, 424]                        # key 的边界
slot_mapping = [                                    # 每个 token 写到 KV cache 的哪个槽位
    # A 的 block 5 的 256 个槽 + block 6 的 84 个槽
    5*256+0, 5*256+1, ..., 5*256+255, 6*256+0, ..., 6*256+83,
    # B 的 block 7 的 84 个槽（block 5 已缓存无需再写）
    7*256+0, ..., 7*256+83
]
block_tables = [[5, 6], [5, 7]]

set_context(is_prefill=True, cu_seqlens_q, cu_seqlens_k, slot_mapping, block_tables)
```

### 3.4 模型前向（Day 3-4，TP 视角）

```
Qwen3ForCausalLM.forward(input_ids=[424], positions=[424])
  │
  ▼
VocabParallelEmbedding                                          ← Day 3-4
  每 rank 持一半 vocab，查不到的输出 0
  all_reduce → [424, 2560]   # 完整 hidden
  │
  ▼
for layer in 28 个 Qwen3DecoderLayer:                           ← Day 3-4 "接力棒"
  │
  ├─ RMSNorm (input_layernorm)
  ├─ Qwen3Attention:
  │    ├─ QKVParallelLinear   # rank 0: heads[0:16]+kv[0:4]
  │    │                      # rank 1: heads[16:32]+kv[4:8]
  │    │    Column 并行 → 各 rank 各算自己的 Q/K/V 子集，无通信
  │    ├─ q_norm / k_norm     # QK-Norm（Qwen3 独有，稳 softmax）
  │    ├─ RotaryEmbedding     # 各 rank 对自己的 Q/K 就地旋转
  │    ├─ Attention:
  │    │    ├─ store_kvcache_kernel (Triton)
  │    │    │    按 slot_mapping 把 K/V 写入 kv_cache[0/1, layer_id, block_id, offset]
  │    │    │    ⚠️ B 的 block 5 对应的槽不在 slot_mapping 里，不重写（它已缓存）
  │    │    └─ flash_attn_varlen_func
  │    │         读 kv_cache 里完整的 K/V（A 刚写的 340 个 + B 共享的 256 个+新的 84 个）
  │    │         按 cu_seqlens 切出每条序列做 attention
  │    └─ RowParallelLinear (o_proj)   # 各 rank 部分结果 → all_reduce 汇合
  │
  ├─ 融合 add_rms_forward (residual + post_attention_layernorm)
  ├─ Qwen3MLP:
  │    ├─ MergedColumnParallelLinear (gate_up_proj)  # Column 并行，各 rank 算一半维
  │    ├─ SiluAndMul                                 # SwiGLU: silu(gate)*up
  │    └─ RowParallelLinear (down_proj)              # all_reduce
  │
  └─ 融合 add_rms 回到下一层的 input_layernorm
  │
  ▼
最终 RMSNorm → hidden[424, 2560]
  │
  ▼
compute_logits：只取每个 seq 最后一个 token 的 hidden
  = hidden[[339, 423]]  → [2, 2560]              # A 的最后一个 + B 的最后一个
  │
  ▼
ParallelLMHead: 每 rank 算一半 vocab → [2, vocab/2]
  all_gather → [2, vocab]                       ← 完整 logits 只在 rank 0 需要
```

**两张卡之间的通信开销**：每个 decoder layer 2 次 all_reduce（o_proj 后、down_proj 后），28 层 = 56 次 all_reduce，加 embedding 1 次 + lm_head 1 次 all_gather。

### 3.5 采样（Day 3-4）

```python
# 仅 rank 0 跑完整采样
Sampler.forward(logits=[2, vocab], temperatures=[0.6, 0.6])
  ├─ logits /= T                              # 温度拉大差异
  ├─ probs = softmax(logits)
  └─ tokens = probs / Exp(1) → argmax         # Gumbel-max trick（无显式随机数）

返回 token_ids = [token_A_new, token_B_new]     # 比如 [3621, 5729]
```

rank 1 的 sampler 返回 None（没用）。

### 3.6 调度后处理（Day 1 + Day 2）

```python
Scheduler.postprocess([A, B], [3621, 5729])
  for seq, tok:
    seq.append_token(tok)        # A.token_ids 变成 341 个，B 变 341 个
    block_manager.may_append(seq)  # 新 token 写入当前最后一块的下一个 offset
                                   # 如果当前块满了 → 计算 hash，试图匹配缓存
    if seq.is_finished:            # EOS or len==max_tokens
       block_manager.deallocate(seq)   # decref 所有块；ref=0 时进 free_block_ids
```

---

## 4. 后续 step：Decode 阶段（Day 1 + Day 5 CUDA Graph）

现在 waiting 空了，走 decode：

```
scheduler.schedule() → prefill 分支空，decode 分支：
  对 running 里每条 seq：调 block_manager.may_append，不行就 preempt 放回 waiting

bs = 2（当前 running 数）
model_runner.call("run", [A, B], is_prefill=False)
  │
  ▼ prepare_decode
  input_ids    = [A.last_token, B.last_token]   # [2]
  positions    = [340, 340]                      # 两条 seq 都到了第 341 个 token
  slot_mapping = [A 新 token 的槽, B 新 token 的槽]
  context_lens = [341, 341]
  block_tables = [[5, 6], [5, 7]]
  │
  ▼ run_model (decode 走 CUDA Graph 路径)
  graph_bs 找到 ≥2 的最小挡位 → bs=2 的 graph
  把实际 input 拷到 graph_vars 固定 buffer，其余填充位：
    slot_mapping.fill_(-1)   ← 防止填充位乱写 KV cache
    context_lens.zero_()
  graph_vars["input_ids"][:2]  = [..., ...]
  graph_vars["slot_mapping"][:2] = [...]
  ...
  graph.replay()   ← 一瞬间把整套 kernel 发射完，Python 端零开销
  │
  ▼ compute_logits → sampler → 新 token
```

**decode 阶段的不同**：
- `flash_attn_with_kvcache` 而非 `flash_attn_varlen_func`（按 block_tables 查 K/V）
- 每个 seq 只贡献 1 个 Q，但要读历史所有 K/V
- 用 CUDA Graph，Python 开销几乎为零

重复上面的循环直到 A、B 都 finish（EOS 或打到 max_tokens=50）。

---

## 5. Prefix Cache 在整个流程里带来了什么

```
没有 prefix cache（naive）                  有 prefix cache（nano-vllm）
─────────────────────                       ────────────────────────────
A 算 340 tokens prefill                     A 算 340 tokens prefill
B 算 340 tokens prefill                     B 只算 84 tokens prefill  ✨
共 680 token 的 attention                   共 424 token 的 attention
                                            KV cache 的 block 5 被 A、B 共用
                                            （ref_count=2，谁都不能释放它）
```

**省了 256/680 ≈ 38% 的 prefill 算力**。这就是为什么实际业务里长系统提示 + RAG 场景能把吞吐拉高 2-5 倍。

---

## 6. 退出清理（Day 5）

```
Python 解释器退出
  atexit.register(self.exit) 触发：
    rank 0: model_runner.call("exit")
      ├─ write_shm("exit", ...) → rank 1 的 loop 接到 "exit" → break
      ├─ self.exit(): shm.close + shm.unlink + dist.destroy_process_group
    rank 0: for p in self.ps: p.join()    ← 等 rank 1 退干净
```

---

## 7. 六视角闭环表（把所有 Day 串成一张网）

| 视角 | 本例中的体现 | 涉及 Day |
|------|-------------|----------|
| **请求层** | A、B 两个 Sequence，各自的 token_ids / block_table / num_cached_tokens | Day 1 |
| **调度层** | waiting → running；B 因 prefix cache 少算 256 tokens；decode 阶段逐步产出 | Day 1, 2 |
| **显存层** | block 5 被 A、B 共享（ref_count=2）；hash_to_block_id 内容寻址；preempt 时释放 | Day 2 |
| **执行层** | prepare_prefill / prepare_decode 打平 batch；CUDA Graph replay；Context 全局传 attention 元数据 | Day 1, 5 |
| **并行层** | QKV/gate_up 走 Column，o_proj/down 走 Row；每层 2 次 all_reduce；lm_head 1 次 all_gather | Day 3-4 |
| **模型层** | 28 层 Qwen3DecoderLayer；QK-Norm；SwiGLU；RoPE；add_rms 融合；flash-attn varlen/with_kvcache | Day 3-4 |
| **运行时** | 主进程 spawn 子进程；NCCL + SharedMemory 双通道；三步启动；atexit 清理 | Day 5 |

---

## 8. 六大加速杠杆在这个例子里各自贡献了什么

1. **Continuous batching**：A、B 同一个 step 里一起 prefill，GPU 不空转
2. **Paged KV + Prefix cache**：B 的前 256 tokens 完全复用 A 写的 block 5
3. **Flash-attn**：attention 的 O(n²) IO 降为 O(n)，对 340-token 的 prefill 影响大
4. **Chunked prefill**：本例 tokens 预算没卡住，但如果 prompt 是 20000 tokens，会切成多个 chunk 混入 decode
5. **CUDA Graph**：decode 阶段每步只传 2 个 token，但要跑 28 层 + 若干 kernel，Graph 让 Python 开销归零
6. **TP=2**：每张卡只扛一半参数 + 一半 KV cache；通信开销 = 每层 2 次 all_reduce，对 4B 模型占比小

---

## 9. 一句话串整个项目

> **nano-vllm 用 Scheduler 编排时间、BlockManager 编排显存、ModelRunner 编排设备、TP 切分编排多卡、CUDA Graph 编排 kernel**——五个编排层叠加，让 1200 行 Python 跑得接近一个成熟推理框架的核心效率。

这就是 Day 1–5 学完后需要内化的心智模型：**同一条请求的生命周期里，每一层抽象都在给上面或下面让渡一些自由度，以换取并发、复用、或 kernel 效率**。

---

## 附录：调用栈速查表

### Prefill 调用栈（TP=2，rank 0 视角）

```
LLM.generate
└─ LLMEngine.step
   ├─ Scheduler.schedule                                        [Day 1, 2]
   │  ├─ BlockManager.can_allocate
   │  └─ BlockManager.allocate    ─ xxhash + hash_to_block_id   [Day 2]
   ├─ ModelRunner.call("run", seqs, True)                       [Day 5]
   │  ├─ write_shm → rank 1 unblocks
   │  └─ self.run
   │     ├─ prepare_prefill (input_ids/positions/slot_mapping)  [Day 1]
   │     ├─ set_context (is_prefill=True)                       [Day 1]
   │     └─ Qwen3ForCausalLM.forward                            [Day 3-4]
   │        ├─ VocabParallelEmbedding + all_reduce              [Day 3-4]
   │        └─ for 28 layers:
   │           ├─ RMSNorm
   │           ├─ QKVParallelLinear (Column)                    [Day 3-4]
   │           ├─ q_norm / k_norm (QK-Norm)                     [Day 3-4]
   │           ├─ RotaryEmbedding                               [Day 3-4]
   │           ├─ store_kvcache_kernel (Triton)                 [Day 3-4]
   │           ├─ flash_attn_varlen_func                        [Day 3-4]
   │           ├─ RowParallelLinear (o_proj) + all_reduce       [Day 3-4]
   │           ├─ add_rms_forward (fused residual + norm)       [Day 3-4]
   │           ├─ MergedColumnParallelLinear (gate_up)          [Day 3-4]
   │           ├─ SiluAndMul                                    [Day 3-4]
   │           └─ RowParallelLinear (down_proj) + all_reduce    [Day 3-4]
   ├─ Sampler.forward (Gumbel-max, rank 0 only)                 [Day 3-4]
   └─ Scheduler.postprocess                                     [Day 1, 2]
      ├─ Sequence.append_token
      └─ BlockManager.may_append / deallocate
```

### Decode 调用栈（差异部分）

```
├─ prepare_decode (slot_mapping 1 个/seq)                       [Day 1]
├─ set_context (is_prefill=False)                               [Day 1]
└─ run_model (CUDA Graph 路径)                                  [Day 5]
   ├─ 找 ≥bs 的 graph 挡位
   ├─ slot_mapping.fill_(-1) + 拷贝实际数据
   ├─ graph.replay()                                            [Day 5]
   └─ flash_attn_with_kvcache (按 block_tables 查 K/V)          [Day 3-4]
```
