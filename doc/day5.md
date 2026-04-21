# Day 5 学习笔记：运行时系统扫尾（多进程 TP、CUDA Graph、显存估算）

> 本日补齐前面没深入的运行时基础设施：多进程 TP 启动与 RPC 通信、三步启动流程（warmup / allocate_kv_cache / capture_cudagraph）、显存公式、配套小文件（config / sampling_params / llm / bench），最后做项目级回顾。

---

## 0. 盘点：到 Day 5 为止还没学到的地方

| 模块 | 文件 / 函数 | 状态 |
|------|-------------|------|
| 多进程 + TP 启动 | `model_runner.py` 的 `__init__` / `loop` / `read_shm` / `write_shm` / `call` | 没讲 |
| CUDA Graph | `warmup_model` / `allocate_kv_cache` / `capture_cudagraph` | 提过没细讲 |
| 显存估算公式 | `allocate_kv_cache` 里那几行 | 没讲 |
| 启动 & 退出流程 | `LLMEngine.__init__` 的多进程起 + `atexit.register(exit)` | 没讲 |
| 小配套文件 | `config.py` / `sampling_params.py` / `llm.py` / `bench.py` | 没系统看 |

Day 5 把这五块一次补齐。

---

## 1. 多进程 + TP 运行时

Day 3 讲了"TP 怎么切权重"，但**"多卡怎么协同运行"从来没讲过**。这一节补齐。

### 1.1 启动拓扑：rank 0 是控制面，其他 rank 是执行面

先看 `LLMEngine.__init__`（`llm_engine.py:17-35`）：

```python
ctx = mp.get_context("spawn")
for i in range(1, config.tensor_parallel_size):
    event = ctx.Event()
    process = ctx.Process(target=ModelRunner, args=(config, i, event))
    process.start()
    self.ps.append(process)
    self.events.append(event)
self.model_runner = ModelRunner(config, 0, self.events)
```

- **rank 0** 是主进程本身——和 `LLMEngine` / `Scheduler` / `tokenizer` 住一起。
- **rank 1..TP-1** 通过 `multiprocessing.Process` 各自 spawn 一个子进程，各自实例化 `ModelRunner(config, rank=i, event)`。
- rank 0 持有所有子进程的 `Event` 列表；子 rank 各持有自己的 `Event`。

### 1.2 每个 ModelRunner 进程做什么

看 `ModelRunner.__init__`（`model_runner.py:17-48`）：

```python
dist.init_process_group("nccl", "tcp://localhost:2333", world_size=..., rank=rank)
torch.cuda.set_device(rank)
self.model = Qwen3ForCausalLM(hf_config)
load_model(self.model, config.model)    # 每卡独立加载（TP 切片）
self.sampler = Sampler()
self.warmup_model()
self.allocate_kv_cache()
if not self.enforce_eager:
    self.capture_cudagraph()

if self.world_size > 1:
    if rank == 0:
        self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
        dist.barrier()
    else:
        dist.barrier()
        self.shm = SharedMemory(name="nanovllm")
        self.loop()    # ← 子 rank 不 return，进入死循环
```

关键点：

- 每张卡都完整走"初始化模型 → 加载权重 → warmup → 分 KV cache →（可选）capture graph"。这五步在所有 rank 对称跑。
- NCCL 初始化通过 `tcp://localhost:2333` 完成，这是 dist 进程组的集结点。
- **共享内存**（POSIX SharedMemory，1MB）用来传 rank 0 → 其他 rank 的方法调用。rank 0 `create=True`，子 rank `create` 默认 `False`（打开已存在的）。`dist.barrier()` 保证 rank 0 先创建好、子 rank 才打开。
- 子 rank 初始化完直接进 `self.loop()`——**永远不返回**，只等待 rank 0 发来的命令。

### 1.3 rank 0 → 其他 rank 的 RPC 机制

看 `model_runner.py:61-89`：

```python
def loop(self):                           # 子 rank 的主循环
    while True:
        method_name, args = self.read_shm()
        self.call(method_name, *args)
        if method_name == "exit":
            break

def read_shm(self):                       # 子 rank 阻塞等 rank 0
    assert self.world_size > 1 and self.rank > 0
    self.event.wait()                     # 阻塞，直到 rank 0 set Event
    n = int.from_bytes(self.shm.buf[0:4], "little")
    method_name, *args = pickle.loads(self.shm.buf[4:n+4])
    self.event.clear()                    # 自己把 Event 清零
    return method_name, args

def write_shm(self, method_name, *args):  # rank 0 写入 + 唤醒所有子 rank
    assert self.world_size > 1 and self.rank == 0
    data = pickle.dumps([method_name, *args])
    n = len(data)
    self.shm.buf[0:4] = n.to_bytes(4, "little")
    self.shm.buf[4:n+4] = data
    for event in self.event:
        event.set()

def call(self, method_name, *args):       # 统一入口
    if self.world_size > 1 and self.rank == 0:
        self.write_shm(method_name, *args)    # rank 0 先广播方法
    method = getattr(self, method_name, None)
    return method(*args)                       # 所有 rank 各自本地执行
```

### 1.4 调用时序举例

以 `engine.step()` 里的 `self.model_runner.call("run", seqs, is_prefill)` 为例（TP=2）：

```
rank 0（主进程）                              rank 1（子进程）
call("run", seqs, is_prefill)                （阻塞在 event.wait()）
  └─ write_shm("run", seqs, is_prefill)
     └─ pickle 后写共享内存
     └─ for event in events: event.set()   ──► 被唤醒
  └─ getattr(self, "run")                          └─ read_shm() 读出 "run" + args
  └─ self.run(seqs, is_prefill)                    └─ self.call("run", seqs, is_prefill)
     ├─ prepare_prefill（各卡同逻辑）                   ├─ getattr + self.run(...)
     ├─ run_model                                      ├─ prepare_prefill
     │   └─ model forward                              ├─ run_model → forward
     │      └─ 层内 all_reduce ◄── NCCL ──► 层内 all_reduce
     ├─ sampler（仅 rank 0 才做真正采样）              ├─ sampler 返回 None
     └─ 返回 token_ids                                 └─ 返回 None（继续 loop）
```

- **方法名和参数**走 `pickle + POSIX SharedMemory`（CPU 侧，慢但小数据够用）。
- **实际前向的张量数据**走 **NCCL**（GPU 侧）。
- 两条通道各司其职：**控制面 CPU、数据面 GPU**。

### 1.5 Sequence 的自定义 pickle

顺带回应一个 Day 1 留的小点：`sequence.py` 有 `__getstate__` / `__setstate__`。为什么？因为 `write_shm` 要 `pickle.dumps(seqs)`，如果直接 pickle 完整 Sequence 对象，`token_ids` 列表很长、pickle 很慢。定制版只传必要字段（`num_tokens` / `num_prompt_tokens` / `num_cached_tokens` / `num_scheduled_tokens` / `block_table` / `last_token_或完整list`），decode 阶段省到只传 `last_token` 一个 int。跨进程通信开销大幅降低。

### 1.6 退出流程

```python
# LLMEngine.__init__
atexit.register(self.exit)

def exit(self):
    self.model_runner.call("exit")   # 触发子 rank 跳出 loop
    del self.model_runner             # rank 0 自己 exit
    for p in self.ps:
        p.join()                      # 等子进程退出

# ModelRunner.exit
def exit(self):
    if self.world_size > 1:
        self.shm.close()
        dist.barrier()
        if self.rank == 0:
            self.shm.unlink()         # 只 rank 0 删共享内存段
    torch.cuda.synchronize()
    dist.destroy_process_group()
```

`atexit.register` 保证 Python 退出时无论正常结束还是异常都会清理子进程和共享内存。

---

## 2. 三步启动：warmup + allocate_kv_cache + capture_cudagraph

这三步按顺序跑一次，决定整个推理的显存布局和执行效率。

### 2.1 `warmup_model`（`model_runner.py:91-101`）

```python
def warmup_model(self):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
    seq_len = min(max_num_batched_tokens, max_model_len)
    num_seqs = min(max_num_batched_tokens // seq_len, self.config.max_num_seqs)
    seqs = [Sequence([0] * seq_len) for _ in range(num_seqs)]
    for seq in seqs:
        seq.num_scheduled_tokens = seq_len
    self.run(seqs, True)                # 跑一次最大规模的 prefill
    torch.cuda.empty_cache()
```

用假数据跑一次**最大 batch 的 prefill**，让 PyTorch 把所有中间 buffer 的峰值显存用出来（`reset_peak_memory_stats` 后记录）。目的是"知道除了 KV cache 之外还要预留多少显存"。

这一步跑完 `torch.cuda.memory_stats()["allocated_bytes.all.peak"]` 就准确了。

### 2.2 `allocate_kv_cache`（`model_runner.py:103-121`）

```python
free, total = torch.cuda.mem_get_info()
used = total - free
peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
num_kv_heads = hf_config.num_key_value_heads // self.world_size
head_dim = getattr(hf_config, "head_dim", ...)
block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * hf_config.dtype.itemsize
config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes
```

**显存公式逐项解读**：

- `total * gpu_memory_utilization` — 我们允许占用的总显存（默认 90%）。
- `- used` — 减掉卡上其他进程已占的（不属于我们的）。
- `- peak` — 减掉我们 warmup 时的峰值（模型权重 + activation buffer + 临时张量）。
- `+ current` — 加回当前已稳定占用的部分（模型权重，warmup 的临时张量已释放）。
- `peak - current` 这个差值 ≈ **激活/临时 buffer 的峰值**，必须预留。

剩下的除以 `block_bytes`（一个 block 的字节数）得到能装几个 KV block。

**block_bytes 公式**：

```
2 (K和V) × num_hidden_layers × block_size × num_kv_heads × head_dim × dtype_bytes
```

Qwen3-0.6B，TP=1，block_size=256，bf16：
`2 × 28 × 256 × 8 × 128 × 2 = 29,360,128 bytes ≈ 28 MB` 每块

如果 `gpu_memory_utilization=0.9` 在 24GB 卡上大概能分到 `(21.6GB - model 1.2GB - peak buffer 几GB) / 28MB ≈ 500+ blocks`。实际取决于卡和 warmup 结果。

然后把 kv_cache 按层绑到每个 Attention 模块：

```python
self.kv_cache = torch.empty(2, num_hidden_layers, num_kvcache_blocks, block_size, num_kv_heads, head_dim)
layer_id = 0
for module in self.model.modules():
    if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
        module.k_cache = self.kv_cache[0, layer_id]
        module.v_cache = self.kv_cache[1, layer_id]
        layer_id += 1
```

每层的 `Attention.k_cache` / `v_cache` 是这个大 tensor 的 **view**，不拷贝。

### 2.3 `capture_cudagraph`（`model_runner.py:223-258`）

```python
max_bs = min(max_num_seqs, 512)
max_num_blocks = (max_model_len + block_size - 1) // block_size
input_ids = torch.zeros(max_bs, dtype=torch.int64)
positions = torch.zeros(max_bs, dtype=torch.int64)
slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
context_lens = torch.zeros(max_bs, dtype=torch.int32)
block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
outputs = torch.zeros(max_bs, hf_config.hidden_size)

self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
self.graphs = {}
self.graph_pool = None

for bs in reversed(self.graph_bs):
    graph = torch.cuda.CUDAGraph()
    set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])
    outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # warmup
    with torch.cuda.graph(graph, self.graph_pool):
        outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # capture
    if self.graph_pool is None:
        self.graph_pool = graph.pool()
    self.graphs[bs] = graph
    torch.cuda.synchronize()
    reset_context()

self.graph_vars = dict(input_ids=..., positions=..., slot_mapping=..., context_lens=..., block_tables=..., outputs=...)
```

**几个关键点**：

**(a) 为什么只 capture decode？**
- decode 每步张量形状几乎不变（bs × 1 token），很适合 Graph。
- prefill 每步 token 总数波动大，Graph 复用率低。

**(b) 为什么选这组 bs 值？**
- `[1, 2, 4, 8] + range(16, 512+1, 16)` 共 36 个离散 bs。
- 实际 decode 时 `next(x for x in graph_bs if x >= bs)` 找到"≥实际 bs 的最小 graph"。比如实际 bs=13 → 用 bs=16 的 graph（前 13 行有效，后 3 行是填充）。
- 离散挡位避免了"每个 bs 都 capture 一次"的爆炸式显存开销。

**(c) graph_pool 是什么？**
- CUDA Graph 默认每个 graph 独占一份显存池。
- 所有 graph 共用一个 `graph_pool` 大幅节省显存——workspace 可以复用。
- 第一个 graph 创建池，之后所有 graph 都挂在同一个池上。

**(d) 为什么 `reversed(graph_bs)`？**
- 从最大 bs 先 capture。最大 bs 的 workspace 最大，后续小 bs 的 workspace 都能塞进去。
- 如果从小往大 capture，小 graph 先占了池，大 graph 时发现池太小要重分。

**(e) capture 前先 warmup 一次**（`outputs[:bs] = ...` 两行）
- 第一次执行会触发 lazy compilation / cache 预热。如果直接 capture 会把编译步骤也录进去。
- 所以每个 bs 先跑一次 eager warmup，再进 `with torch.cuda.graph(...)` block 正式 capture。

**(f) 执行时怎么 replay？**（`run_model` 中，`model_runner.py:200-213`）

```python
bs = input_ids.size(0)
graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
graph_vars = self.graph_vars
graph_vars["input_ids"][:bs] = input_ids
graph_vars["positions"][:bs] = positions
graph_vars["slot_mapping"].fill_(-1)
graph_vars["slot_mapping"][:bs] = context.slot_mapping
graph_vars["context_lens"].zero_()
graph_vars["context_lens"][:bs] = context.context_lens
graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
graph.replay()
return self.model.compute_logits(graph_vars["outputs"][:bs])
```

- 把实际数据拷贝到 graph 当初 capture 时用的那几个固定 tensor（`input_ids`、`positions` 等）。
- `slot_mapping.fill_(-1)` 很重要——多出来的槽位填 -1，`store_kvcache_kernel` 里 `if slot == -1: return` 会跳过这些填充位，不往 KV cache 乱写。
- `graph.replay()` 一瞬间把整套 kernel launch 完——省掉 Python 端的 for-loop、各种 shape 检查、调度开销。对于纯 kernel 的 decode batch，这能带来可观加速。

---

## 3. 小配套文件扫尾

### 3.1 `config.py`（26 行）

```python
@dataclass(slots=True)
class Config:
    model: str                           # 模型路径
    max_num_batched_tokens: int = 16384  # 每 step prefill token 预算
    max_num_seqs: int = 512              # running 队列最大序列数
    max_model_len: int = 4096            # 最长序列长度
    gpu_memory_utilization: float = 0.9  # 可用显存比例
    tensor_parallel_size: int = 1
    enforce_eager: bool = False          # True 不 capture CUDA Graph
    hf_config: AutoConfig | None = None
    eos: int = -1                        # 由 LLMEngine 填充
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1         # 由 allocate_kv_cache 填充

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0   # 目前只支持 256 的倍数
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
```

**几个调优旋钮**：
- `max_num_batched_tokens` 大 → prefill 吞吐高，但 warmup 峰值显存大，KV block 就少。
- `max_num_seqs` 大 → 并发高，但 CUDA Graph 要 capture 更多挡位，启动慢。
- `gpu_memory_utilization` 调低 → 留更多显存给其他进程，但 KV block 少、容易触发 preempt。
- `enforce_eager=True` → 跳过 Graph，启动快、内存省，但 decode 慢（每步都 launch kernel）。

### 3.2 `sampling_params.py`（11 行）

```python
@dataclass(slots=True)
class SamplingParams:
    temperature: float = 1.0
    max_tokens: int = 64
    ignore_eos: bool = False

    def __post_init__(self):
        assert self.temperature > 1e-10, "greedy sampling is not permitted"
```

极简。只有 3 个字段：温度、最大生成长度、是否忽略 EOS（bench 时用，避免中途提前结束）。注意 `assert temperature > 1e-10`——不支持 T=0 贪心，因为 Sampler 里会除 T。要贪心得用极小的 T（比如 1e-5）近似。

### 3.3 `llm.py`（4 行）

```python
from nanovllm.engine.llm_engine import LLMEngine

class LLM(LLMEngine):
    pass
```

**空壳**。只为了让 `from nanovllm import LLM` 这种导入符合 vLLM 社区习惯（`from vllm import LLM`）。所有逻辑都在 `LLMEngine`。

### 3.4 `bench.py`（33 行）

```python
num_seqs = 256
max_input_len = 1024
max_ouput_len = 1024

prompt_token_ids = [[randint(0, 10000) for _ in range(randint(100, max_input_len))] for _ in range(num_seqs)]
sampling_params = [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=randint(100, max_ouput_len)) for _ in range(num_seqs)]

llm.generate(["Benchmark: "], SamplingParams())    # 第一次 warmup
t = time.time()
llm.generate(prompt_token_ids, sampling_params, use_tqdm=False)
t = (time.time() - t)
throughput = total_tokens / t
```

**几个值得注意的细节**：

- 256 条随机长度 prompt + 随机长度输出——模拟真实负载的长短混合。
- `ignore_eos=True`——强制跑满 `max_tokens`，否则随机 prompt 可能很快产 EOS，测不准。
- 第一次 `llm.generate(["Benchmark: "], ...)` 是 warmup——触发第一次 forward 的 lazy 初始化（tokenizer cache、CUDA 内核 JIT 编译等），避免污染后续测时。
- `use_tqdm=False`——关掉进度条避免时间干扰。

---

## 4. 项目级回顾

### 4.1 六视角总结

| 视角 | 职责 | 代码落点 | 去掉它会怎样 |
|------|------|----------|--------------|
| **请求层** | 表示 prompt + 运行态 | `Sequence` / `SamplingParams` | 无法区分不同请求 |
| **调度层** | 决定跑谁、何时抢占 | `Scheduler` | 只能串行一条一条推，吞吐骤降 |
| **显存层** | paged KV + prefix cache | `BlockManager` | KV 显存碎片，无法共享前缀 |
| **执行层** | 张量准备 + 前向 | `ModelRunner` + `Context` | 无法打平 batch，无法 CUDA Graph |
| **并行层** | 多卡 TP + 通信 | `layers/linear+embed_head` + `loader.py` + 共享内存 RPC | 单卡 OOM，跑不了大模型 |
| **模型层** | 网络结构 | `qwen3.py` + `layers/` 其他 | 换模型就没了 |

### 4.2 "为什么 nano-vllm 比 naive 推理快" 六大杠杆

1. **Continuous batching**：不同序列在同一 step 里打包计算，GPU 利用率高。
2. **Paged KV cache + prefix cache**：KV 显存碎片为零，相同前缀复用。
3. **Flash-attention**：attention 的 IO 和 kernel 合并，减少显存往返。
4. **Chunked prefill**：长 prompt 切片推进，不阻塞 decode 吞吐。
5. **CUDA Graph**：decode 阶段消除 Python / kernel launch 开销。
6. **Tensor parallel**：大模型显存压力摊到多卡，同时通过 Column→Row 配对把通信降到每层 2 次 all_reduce。

### 4.3 这个项目相比真实 vLLM 缺什么

- 只支持 Qwen3，其他模型要各自实现 `models/xxx.py`。
- 没有 pipeline parallel / expert parallel，只有 tensor parallel。
- 没有 speculative decoding / guided decoding / JSON schema 约束。
- 没有 top-k / top-p / min-p 采样，只有温度采样。
- 没有 LRU 清理 `hash_to_block_id`——僵尸块表无限增长。
- 没有 request 级的 API server（OpenAI-compatible HTTP），只有 `llm.generate()` 批量接口。
- 没有量化支持（AWQ/GPTQ/FP8）。
- CUDA Graph 只 capture 固定 bs 挡位，不 capture prefill 的变长 token。

但正因为砍掉这些，nano-vllm 才只有 ~1200 行——**足以作为理解推理框架核心抽象的最短教材**。

### 4.4 面向后续学习的指路

读完 nano-vllm 后可以顺这条线走：

- **vLLM 本体**：看它的 `LLMEngine` / `scheduler` / `block_manager` 怎么扩展到 8 种模型、5 种量化、流式 API、spec decoding。重点看 vLLM v1 的 `uniproc_executor` / `multiproc_executor` / `ray_distributed_executor` 不同执行器。
- **SGLang**：RadixAttention 怎么把 prefix cache 做成 trie；RuntimeCache 管理；front-end DSL。
- **TensorRT-LLM**：纯 C++ + kernel fusion，观察它的 attention plugin、KV cache manager、masked_multihead_attention kernel。
- **DeepSpeed-Inference / FasterTransformer**：更底层的 kernel 实现，理解 attention 融合的各种技巧。

---

## 5. Day 5 压缩 5 句话

1. **多进程 TP**：rank 0 主进程控制面，rank 1+ 子进程执行面；方法调用走 `pickle + SharedMemory`，张量数据走 NCCL。
2. **三步启动**：`warmup` 探峰值 → `allocate_kv_cache` 按剩余显存分 block → `capture_cudagraph` 预录 36 个 bs 挡位的 decode graph。
3. **显存公式**：`(total·util - 其他占用 - 峰值 + 当前稳定) / block_bytes`，把"激活和临时 buffer 的峰差"预留出来。
4. **CUDA Graph** 只 capture decode（固定 shape 复用率高）、用 `graph_pool` 共享 workspace、从最大 bs 开始录、replay 时 `slot_mapping.fill_(-1)` 防止填充位乱写。
5. **配置调优三杠杆**：`max_num_batched_tokens` 控 prefill 吞吐 vs KV 预算、`max_num_seqs` 控并发 vs graph 启动时间、`gpu_memory_utilization` 控显存预留 vs preempt 频率。

---

## 6. 五日学习闭环

```
Day 1 : generate() 调用栈 + 数据流动                 （上层：怎么跑完一个请求）
Day 2 : BlockManager 分配/回收 + prefix cache        （显存：KV 怎么管）
Day 3-4 : Qwen3 模型 + layers/ + TP 权重加载         （模型：算子和并行）
Day 5 : 多进程 RPC + CUDA Graph + 显存估算 + 扫尾    （运行时：启动和调度基础设施）
```

至此 nano-vllm 的 ~1200 行代码被拆解成**六视角**、**六杠杆**、**五日**——每个抽象的动机、实现和取舍都有迹可循。后续读真实 vLLM / SGLang / TRT-LLM 时，这套框架仍然适用，只是每一格里的实现更复杂。
