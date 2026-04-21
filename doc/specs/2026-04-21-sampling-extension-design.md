# Sampling Extension (top-k / top-p / greedy / repetition_penalty) · 设计文档

- **日期**：2026-04-21
- **作者**：Orange-Bei（与 Claude brainstorming）
- **状态**：Draft，待评审

---

## 1. 背景

Nano-vLLM 当前的 `Sampler`（`nanovllm/layers/sampler.py`）只支持 `temperature` 缩放 + Gumbel-max 采样，且 `SamplingParams.__post_init__` 通过 `assert self.temperature > 1e-10` 显式禁止了 greedy 解码。在实际推理应用中，`top_k`、`top_p`、`repetition_penalty`、greedy 是最常用的采样维度，缺失这些特性限制了本项目在真实场景的可用性，也阻碍了后续用采样策略对比性能的工作。

改造涉及三个技术约束：

- `Sampler.forward` 有 `@torch.compile` 装饰器，新增控制流若引入 Python 分支会触发 recompile
- 当前 Sampler **不在 CUDA graph 内**（graph 只捕获到模型 forward 的 hidden state，`compute_logits` 与 sampler 都是 eager），改造应保持这一边界不变
- `repetition_penalty` 需要每 seq 的可变长度历史 token，本质 ragged，与 `@torch.compile` 要求的形状稳定性天然冲突

本设计是二次开发 roadmap 第 2 项，复用第 1 项（引擎指标）作为性能评估尺子。

## 2. 目标与非目标

### 2.1 目标

- **支持 4 种采样特性**：greedy（`temperature=0`）、`top_k`、`top_p`、`repetition_penalty`
- **mixed batch**：同一个 decode batch 内不同 seq 可携带不同的采样参数（包括 greedy 和 非 greedy 混合）
- **向后兼容**：`SamplingParams(temperature=0.6, max_tokens=256)` 等既有调用零改动；默认参数行为与当前完全一致（除计算路径差异）
- **编译稳定性**：`Sampler.forward` 保留 `@torch.compile`，通过形状稳定 + `torch.where` 消除数据相关的 Python 分支，避免 recompile 爆炸
- **性能不显著回归**：`bench.py` 默认参数下吞吐回归 ≤ 5%

### 2.2 非目标（YAGNI）

- 不做 `min_p`、`presence_penalty`、`frequency_penalty`、`logit_bias`、per-request `seed` 等额外开关
- 不把 Sampler 拉进 CUDA graph（留 eager 外部路径，避免预分配 `[B, V]` 级张量）
- 不做 `repetition_penalty` 历史去重优化（scatter 幂等性保证正确性，优化留给性能回归触发时再做）
- 不做跨 TP rank 协调（Sampler 只在 rank 0 执行，不变）
- 不做输出分布的跨硬件一致性验证（项目运行在 3×4090 单环境）

## 3. 架构决策

采用**两阶段**架构：

1. **Eager 预处理**：`apply_repetition_penalty(logits, seqs)`——Python 循环，per-seq 做 gather / scatter。处理 ragged 的历史 token
2. **编译核心**：`Sampler.forward(logits, temperatures, top_k, top_p)`——`@torch.compile`，形状全部稳定 `[B]` / `[B, V]`，控制流由 `torch.where` 完成

### 3.1 关键设计选择

| 决策 | 选择 | 理由 |
|---|---|---|
| Sampler 与 CUDA graph 的关系 | 维持现状，Sampler 在 graph 外 | 拉进 graph 需要固定所有采样张量形状 + 重写 `capture_cudagraph`，复杂度不成比例 |
| greedy 在 mixed batch 中的实现 | Gumbel 路径与 argmax 路径都跑，逐行 `torch.where(temperature<EPS, argmax, sampled)` 选 | 单一编译图稳定；greedy 行多算一次 argmax 的开销可忽略（V≈150k，但 sort 已经 O(V log V) 占主要） |
| `repetition_penalty` 的历史范围 | `prompt + completion`（整个 `seq.token_ids`） | 对齐 vLLM 与 HF Transformers 行为 |
| 禁用状态的 sentinel | `top_k=-1`、`top_p=1.0`、`repetition_penalty=1.0`、`temperature=0.0` | 向量化 no-op 等价，避免 Python 分支 |
| 运行时校验 | 只在 `SamplingParams.__post_init__` 做 assert，运行时信任输入 | 失败得早；编译核心里不加 assert |

### 3.2 侵入性评估

- **不碰** `nanovllm/engine/model_runner.py` 的 CUDA graph 捕获逻辑（只改 `prepare_sample` 扩字段 + `run` 串接预处理）
- **不碰** `block_manager.py` / `scheduler.py` / `metrics.py`
- **不碰** TP worker 的 pickle IPC 路径（`Sequence.__getstate__` 不变，新字段只在 rank 0 用）

## 4. 核心数据结构与算法

### 4.1 `SamplingParams` 扩展

```python
@dataclass(slots=True)
class SamplingParams:
    temperature: float = 1.0
    top_k: int = -1                       # -1 = 禁用
    top_p: float = 1.0                    # 1.0 = 禁用
    repetition_penalty: float = 1.0       # 1.0 = 禁用
    max_tokens: int = 64
    ignore_eos: bool = False

    def __post_init__(self):
        assert self.temperature >= 0.0, f"temperature must be >= 0, got {self.temperature}"
        assert self.top_k == -1 or self.top_k >= 1, f"top_k must be -1 (disabled) or >= 1, got {self.top_k}"
        assert 0.0 < self.top_p <= 1.0, f"top_p must be in (0, 1], got {self.top_p}"
        assert self.repetition_penalty > 0.0, f"repetition_penalty must be > 0, got {self.repetition_penalty}"
        assert self.max_tokens > 0
```

- `temperature=0.0` 现在合法（进入 greedy 路径）
- `top_k=0` 拒绝（避免与"-1 禁用"混淆；vLLM 语义对齐，单一 sentinel）
- `top_p=0` 拒绝（无实际意义）

### 4.2 `Sequence` 扩展

在 `__init__` 里从 `sampling_params` 拷出：

```python
self.top_k = sampling_params.top_k
self.top_p = sampling_params.top_p
self.repetition_penalty = sampling_params.repetition_penalty
```

**不进 `__getstate__`**——Sampler 只在 rank 0，TP worker 的 pickle payload 保持不变。

### 4.3 `apply_repetition_penalty`

位置：`nanovllm/layers/sampler.py`（与 `Sampler` 同文件，都是采样相关）

```python
def apply_repetition_penalty(logits: torch.Tensor, seqs: list[Sequence]) -> None:
    """In-place 应用 repetition_penalty。历史范围 = prompt + completion。"""
    for i, seq in enumerate(seqs):
        rp = seq.repetition_penalty
        if rp == 1.0:
            continue
        tokens = torch.tensor(seq.token_ids, device=logits.device, dtype=torch.int64)
        selected = logits[i].gather(0, tokens)
        penalized = torch.where(selected < 0, selected * rp, selected / rp)
        logits[i].scatter_(0, tokens, penalized)
```

- **不 `@torch.compile`**：per-seq token 数量 ragged，不适合编译
- `rp == 1.0` 快路：对最常见情况零开销
- Scatter 幂等：同一 token 在 history 中重复出现不会叠加惩罚

### 4.4 `Sampler.forward` 改造

```python
class Sampler(nn.Module):
    EPS = 1e-5

    @torch.compile
    def forward(self, logits, temperatures, top_k, top_p):
        logits = logits.float()
        V = logits.size(-1)

        # Greedy 路径：直接在原 logits 上 argmax
        greedy_tokens = logits.argmax(dim=-1)

        # top_k：用一次 sort 拿到每行第 k 大的阈值
        k = torch.where(top_k <= 0, V, top_k).clamp(max=V)           # -1 -> V (no-op)
        sorted_logits, sorted_idx = logits.sort(dim=-1, descending=True)
        thresholds = sorted_logits.gather(-1, (k - 1).unsqueeze(-1))
        masked = logits.masked_fill(logits < thresholds, float('-inf'))

        # top_p：复用 sorted_logits 做 nucleus 裁剪
        sorted_probs = sorted_logits.softmax(dim=-1)
        cumsum = sorted_probs.cumsum(dim=-1)
        remove_sorted = (cumsum - sorted_probs) > top_p.unsqueeze(-1)  # top_p=1.0 -> 全 False
        remove = torch.zeros_like(remove_sorted).scatter_(-1, sorted_idx, remove_sorted)
        masked = masked.masked_fill(remove, float('-inf'))

        # temperature + Gumbel-max
        scaled = masked.div(temperatures.clamp(min=self.EPS).unsqueeze(-1))
        probs = scaled.softmax(dim=-1)
        sampled_tokens = probs.div_(
            torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)
        ).argmax(dim=-1)

        # 逐行选：greedy seq 取 argmax，其他取 Gumbel 结果
        return torch.where(temperatures < self.EPS, greedy_tokens, sampled_tokens)
```

**设计要点**：

- **只 sort 一次**：`top_k` 要找第 k 大、`top_p` 要按降序累加概率，两者复用同一个 `sorted_logits`，避免两次 O(V log V)
- **禁用语义即向量化 no-op**：`top_k=-1 → k=V`（阈值是最小值，掩码为空）、`top_p=1.0`（`cumsum - sorted_probs ≤ 1.0` 恒成立）
- **greedy 走原 logits**：不受 `top_k` / `top_p` 影响，对齐 vLLM 语义
- **`temperatures.clamp(min=EPS)` 防除零**：greedy seq 的 `sampled_tokens` 会被 `where` 丢弃，数值不关心
- **编译稳定性**：全是 `torch.where` / 张量运算，无 Python `if`；batch size 的动态形状靠 `@torch.compile` 自动泛化（和现状相同）

### 4.5 边界验证

| 组合 | 结果 |
|---|---|
| 新特性全禁用 `top_k=-1, top_p=1.0, rp=1.0`，`temperature>0` | 等价于现状 Sampler（top_k / top_p / rp 全 no-op，走 Gumbel） |
| `temperature=0` | 进入 greedy，输出等价于 `logits.argmax(-1)` |
| `top_k=1` 且 `temperature>0` | `k=1` 时只有最大 logit 存活，Gumbel 和 argmax 等价 |
| `top_p=0`（被 assert 拦截） | 不会发生 |
| `top_p` 极小 | `(cumsum - sorted_probs) > small` 在 position 0 为 False，至少保留最高概率 token，退化为 greedy |
| mixed batch | 每行独立计算阈值 / 掩码 / 采样，`where` 按行选 |

## 5. 实现位置

### 5.1 `nanovllm/sampling_params.py`

替换 `__post_init__`，新增 3 个字段（见 §4.1）。

### 5.2 `nanovllm/engine/sequence.py`

`__init__` 中新增 3 个字段赋值；`__getstate__` / `__setstate__` 不变。

### 5.3 `nanovllm/layers/sampler.py`

- 新增 `apply_repetition_penalty(logits, seqs)` 模块级函数
- 改造 `Sampler.forward`：签名变为 `(logits, temperatures, top_k, top_p)`，新增 `EPS` 类常量

### 5.4 `nanovllm/engine/model_runner.py`

改两处：

`prepare_sample` 扩展为返回 4 个张量：

```python
def prepare_sample(self, seqs):
    temperatures = torch.tensor([s.temperature for s in seqs],
                                dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
    top_k = torch.tensor([s.top_k for s in seqs],
                         dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
    top_p = torch.tensor([s.top_p for s in seqs],
                         dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
    return temperatures, top_k, top_p
```

`run` 改为先 eager apply rep_penalty 再调编译核心。**关键：`run` 需要 `@torch.inference_mode()` 装饰器**——因为 `run_model` 在 inference_mode 里产出 logits（"inference tensor"），而 `apply_repetition_penalty` 的 `scatter_` 需要在 inference_mode 内对 inference tensor 做 in-place 修改，否则 torch 会抛 `Inplace update to inference tensor outside InferenceMode is not allowed`。

```python
@torch.inference_mode()
def run(self, seqs, is_prefill):
    input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
    sample_inputs = self.prepare_sample(seqs) if self.rank == 0 else None
    logits = self.run_model(input_ids, positions, is_prefill)
    if self.rank == 0:
        apply_repetition_penalty(logits, seqs)
        token_ids = self.sampler(logits, *sample_inputs).tolist()
    else:
        token_ids = None
    reset_context()
    return token_ids
```

（`apply_repetition_penalty` 从 `nanovllm.layers.sampler` import）

## 6. API 与对接

### 6.1 用户代码示例

```python
# 向后兼容：现有代码零改动
sampling_params = SamplingParams(temperature=0.6, max_tokens=256)

# 新特性：greedy
sampling_params = SamplingParams(temperature=0.0, max_tokens=256)

# 新特性：top-k + top-p + repetition_penalty
sampling_params = SamplingParams(
    temperature=0.7,
    top_k=50,
    top_p=0.9,
    repetition_penalty=1.1,
    max_tokens=256,
)

# mixed batch
batch = [
    SamplingParams(temperature=0.0),                         # greedy
    SamplingParams(temperature=0.7, top_k=50),                # top-k sampling
    SamplingParams(temperature=1.0, top_p=0.9),               # nucleus sampling
    SamplingParams(temperature=0.8, repetition_penalty=1.2),  # with penalty
]
outputs = llm.generate(prompts, batch)
```

### 6.2 `example.py` / `bench.py` 变更

- `example.py`：保持不变；可选增加一行演示 greedy 的用法
- `bench.py`：保持 `SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=...)`，作为性能回归基准用例

## 7. 边界情况

| 情况 | 处理 |
|---|---|
| `SamplingParams()` 默认 | `top_k=-1, top_p=1.0, rp=1.0, temperature=1.0`，所有新特性禁用，等价于当前 Sampler 行为 |
| 全 batch 都是 greedy | `where` 选到 `argmax_tokens`；`sampled_tokens` 计算白白浪费，可接受（sampler 占端到端 <1%） |
| `top_k` 超过 V | `clamp(max=V)` 保护，等价于无 top_k |
| `top_p` 非常小 | 至少保留 position 0（最大 logit），退化为 greedy |
| rp 历史为空（prompt 阶段首个 token） | 不可能——decode 阶段必然 `len(seq.token_ids) >= 1` |
| 同一 token 在历史中重复 | Scatter 幂等，结果正确 |
| CUDA graph decode | 不受影响：graph 只捕获模型 forward 到 hidden state，Sampler 和新的 rep_penalty 都在 graph 外 |
| TP `world_size > 1` | 仅 rank 0 采样，worker 不变；`Sequence` 新字段不进 `__getstate__` |
| `@torch.compile` 动态 batch size | 与当前 `temperatures[B]` 行为相同，首次编译后自动泛化动态形状 |
| prefill 阶段调用 Sampler | 当前架构每次 `run` 都采样一个 token（prefill 也要 sample 出下一个），新 Sampler 照常工作 |

## 8. 验证

### 8.1 单元测试：`tests/test_sampling_params.py`（无 GPU）

- 默认构造不抛
- 非法输入分别抛 `AssertionError`：`temperature=-1`、`top_k=0`、`top_k=-2`、`top_p=0`、`top_p=1.5`、`repetition_penalty=0`、`repetition_penalty=-1`、`max_tokens=0`
- `SamplingParams(temperature=0)` 合法（旧 assert 已移除）

### 8.2 单元测试：`tests/test_sampler_kernel.py`（GPU，合成 logits）

用 `pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), ...)` 保护。

| 用例 | 断言 |
|---|---|
| 全 `temperature=0` | `sample == logits.argmax(-1)` |
| 默认 `temp=0.6, k=-1, p=1.0, rp=1.0` | 多次采样的经验分布与 `softmax(logits/temp)` 的 χ² 误差 < 阈值 |
| `top_k=1` 非 greedy | `sample == logits.argmax(-1)` |
| `top_k=5` | 1000 次采样的 token 集合 ⊆ 原 logits 的 top-5 |
| `top_p=0.5` 且有单一 token 占 60% 概率 | 所有采样结果都是该 token |
| mixed batch：行 0 greedy、行 1 sampled | 行 0 恒等 argmax，行 1 的分布接近 softmax |

### 8.3 单元测试：`tests/test_apply_repetition_penalty.py`（GPU）

- `rp=1.0`：`torch.allclose(before, after)`
- `rp=2.0`：正 logit 除以 2、负 logit 乘以 2
- Token X 在 history 中重复 5 次：应用后 `logits[i, X]` 等于单次惩罚结果
- Mixed batch：仅 `rp != 1.0` 的行被修改

### 8.4 端到端冒烟：`tests/test_sampling_e2e.py`

沿用 `NANO_VLLM_TEST_MODEL` 环境变量 + `pytest.mark.skipif` 模式。

- `temperature=0`：同一 prompt 两次调用输出完全一致（确定性）
- `top_k=1`：同一 prompt 两次调用输出完全一致（等价 greedy）
- `repetition_penalty=1.5`：对一个容易重复的 prompt，输出 token 集合的去重率高于 `rp=1.0`
- **mixed `SamplingParams` list**：传含 greedy + top_k + top_p + rp 各一条的 list，`generate` 不崩溃 + 每条 `output["metrics"]` 非空
- **无 recompile 爆炸**：`torch._dynamo.reset()` 后，连续 5 次 `generate` 使用不同 batch 大小，结束后读 `torch._dynamo.utils.counters["stats"]["unique_graphs"]` ≤ 一个小常数（例如 ≤ 3）。若该 API 在当前 torch 版本不存在，退化为运行时无异常的观察性断言

### 8.5 性能回归（手工，复用 #1 引擎指标）

`bench.py` 在**默认 `SamplingParams()`** 下跑改造前后两次，对比：

- `EngineMetrics.decode_throughput`：回归 ≤ 5%
- `EngineMetrics.tpot.p50` / `.p99`：回归 ≤ 5%
- `EngineMetrics.ttft.p50`：回归 ≤ 5%

阈值不自动断言，落到 plan 阶段作为验收项人工检查。主要风险点：每 decode step 都会 O(V log V) sort 一次 logits（V≈150k，batch=512 时），这是新引入的固定成本。

## 9. 文件清单

**新建**

- `tests/test_sampling_params.py`（~30 行）
- `tests/test_sampler_kernel.py`（~120 行）
- `tests/test_apply_repetition_penalty.py`（~60 行）
- `tests/test_sampling_e2e.py`（~80 行）
- `tests/conftest.py`（~15 行）—— session 级 `llm` fixture，避免两个 e2e 文件各自 `dist.init_process_group` 冲突

**修改**

- `nanovllm/sampling_params.py`：+3 字段 + 重写 `__post_init__`
- `nanovllm/engine/sequence.py`：`__init__` 中 +3 字段赋值
- `nanovllm/layers/sampler.py`：新增 `apply_repetition_penalty`，改造 `Sampler.forward`
- `nanovllm/engine/model_runner.py`：`prepare_sample` 扩展 + `run` 串接 rep_penalty 调用 + 给 `run` 加 `@torch.inference_mode()`
- `nanovllm/engine/llm_engine.py`：`exit()` 幂等化（fixture 显式 exit 后避免 atexit 二次触发报错）
- `tests/test_metrics_e2e.py`：移除本地 module 级 `llm` fixture，改用 conftest 的 session 级 fixture

**明确不改**

- `nanovllm/engine/scheduler.py`
- `nanovllm/engine/block_manager.py`
- `nanovllm/engine/metrics.py`
- `nanovllm/engine/llm_engine.py` / `nanovllm/llm.py`
- `nanovllm/models/**` / `nanovllm/utils/**`
- `nanovllm/engine/model_runner.py` 的 `capture_cudagraph` 路径
- `Sequence.__getstate__` / `__setstate__`

## 10. 开销分析

- `apply_repetition_penalty`：对 `rp != 1.0` 的 seq 每步 2 个小 GPU op（gather + scatter）。最常见情况（全默认 `rp=1.0`）直接 `continue`，开销为 batch 遍历 + 浮点比较
- `Sampler.forward`：新增一次 `torch.sort` on `[B, V]`。V≈150k、B≤512，理论吞吐在 A100/4090 上是 ~几百微秒级。相对单次 decode step 毫秒级延迟，预估回归 2-5%
- `prepare_sample`：多建 2 个 `[B]` 张量 + `.cuda(non_blocking=True)`，微秒级

## 11. 对后续项目的铺垫

- **chunked prefill 推广（#3）**：本改动不碰 scheduler / chunked prefill 路径，#3 无需考虑采样扩展的交互
- **DP 并行（#4）**：Sampler 继续只在 rank 0 执行，DP 协调层若复用单副本 Sampler 语义即可；无新增跨副本同步需求
