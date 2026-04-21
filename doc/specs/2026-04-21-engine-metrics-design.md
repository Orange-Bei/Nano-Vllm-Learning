# Engine Metrics (TTFT / TPOT / etc.) · 设计文档

- **日期**：2026-04-21
- **作者**：Orange-Bei（与 Claude brainstorming）
- **状态**：Draft，待评审

---

## 1. 背景

Nano-vLLM 当前只在 `LLMEngine.generate()` 的进度条和 `bench.py` 打印一行聚合吞吐（prefill tok/s、decode tok/s）。缺乏：

- 单请求首 token 延迟（TTFT）、每 token 延迟（TPOT）、端到端延迟
- 排队时间 / prefill 时间 / decode 时间的分段归因
- ITL（inter-token latency）分布，用于评估逐 token 体验流畅性
- KV cache 利用率时序、抢占事件计数等调度健康度观测

本项目后续还有采样方式扩展、chunked prefill 推广、DP 并行三个独立改动。每一项都需要"尺子"来衡量改动效果。本设计作为整个二次开发序列的第一步，提供通用可观测性基础设施。

## 2. 目标与非目标

### 2.1 目标

- **per-request 指标**：`arrival_time`、`first_scheduled_time`、`first_token_time`、`finish_time`、`token_times[]`（ITL 全序列）、`preemption_count`；派生 TTFT / TPOT / E2E / queue / prefill / decode time
- **引擎级指标**：KV cache 利用率时序、抢占事件时间轴、每 step 的 batch 占满率、聚合吞吐
- **API**：
  - `LLM.generate()` 返回的每个 dict 增加 `metrics` 字段（`RequestMetrics` 对象）
  - 新增 `engine.get_aggregate_metrics() -> EngineMetrics`（含 p50/p90/p99）
  - 新增 `engine.reset_metrics()`
- **bench.py**：打印结构化汇总表；可选 JSON 导出

### 2.2 非目标（YAGNI）

- 不接入 Prometheus / OpenTelemetry
- 不做 matplotlib 可视化（提供 JSON，由用户自行绘图）
- 不做 GPU 计算时间 vs CPU 调度时间的细粒度剥离（用 `perf_counter` 反映用户感知 wall clock）
- 不做跨 TP rank 的分布式指标采集（所有埋点仅在 rank 0 引擎进程，KV cache 本身也只在 rank 0 的 `BlockManager`）

## 3. 架构决策

采用**混合式**架构：

- **per-seq 细粒度时间戳** 放在 `Sequence` 上（纯 primitive 字段，开销接近 0）
- **全局时序与事件数据** 放在独立的 `MetricsCollector`（KV 利用率时序、抢占事件、step 级样本）
- `get_aggregate_metrics()` 从两处拉数据，组装成统一的 `EngineMetrics`

理由：
- per-seq 数据本来就跟随 `Sequence` 生命周期，就近存储最省事
- 全局数据无自然归属，集中到 collector 里便于统一查询/导出/重置
- collector 是独立对象，未来加新指标只改一个文件

## 4. 核心数据结构

新增 `nanovllm/engine/metrics.py`。

### 4.1 `Percentiles`

```python
@dataclass
class Percentiles:
    avg: float
    min: float
    p50: float
    p90: float
    p95: float
    p99: float
    max: float

    @classmethod
    def from_samples(cls, samples: list[float]) -> "Percentiles": ...
```

空样本时所有字段为 `float("nan")`。

### 4.2 `RequestMetrics`

```python
@dataclass
class RequestMetrics:
    # 原始时间戳 (perf_counter 秒)
    arrival_time: float
    first_scheduled_time: float
    first_token_time: float
    finish_time: float
    token_times: list[float]          # 每个 completion token 落地时刻

    # 统计
    num_prompt_tokens: int
    num_completion_tokens: int
    preemption_count: int

    # 派生指标
    @property
    def ttft(self) -> float:
        return self.first_token_time - self.arrival_time

    @property
    def tpot(self) -> float:
        # 首 token 后的平均每 token 间隔；n-1 作分母
        n = self.num_completion_tokens
        if n <= 1:
            return 0.0
        return (self.finish_time - self.first_token_time) / (n - 1)

    @property
    def e2e_latency(self) -> float:
        return self.finish_time - self.arrival_time

    @property
    def queue_time(self) -> float:
        return self.first_scheduled_time - self.arrival_time

    @property
    def prefill_time(self) -> float:
        return self.first_token_time - self.first_scheduled_time

    @property
    def decode_time(self) -> float:
        return self.finish_time - self.first_token_time

    @property
    def inter_token_intervals(self) -> list[float]:
        return [b - a for a, b in zip(self.token_times, self.token_times[1:])]
```

### 4.3 `StepSample`

```python
@dataclass
class StepSample:
    timestamp: float          # step 开始时刻
    is_prefill: bool
    num_seqs: int
    num_batched_tokens: int
    num_free_blocks: int
    num_used_blocks: int
    step_duration: float      # 该 step 耗时
```

### 4.4 `EngineMetrics`

```python
@dataclass
class EngineMetrics:
    # per-request 分位数
    ttft: Percentiles
    tpot: Percentiles
    e2e_latency: Percentiles
    queue_time: Percentiles
    prefill_time: Percentiles
    decode_time: Percentiles
    itl: Percentiles                  # 所有请求 ITL 展平

    # 全局计数
    total_requests: int
    total_preemptions: int
    prefill_throughput: float
    decode_throughput: float
    wall_clock_seconds: float         # 从最早 arrival_time 到最晚 finish_time

    # 原始时序
    step_samples: list[StepSample]

    def to_dict(self) -> dict: ...
    def summary_table(self) -> str: ...
```

### 4.5 `MetricsCollector`

```python
class MetricsCollector:
    def __init__(self) -> None:
        self.completed_requests: dict[int, RequestMetrics] = {}
        self.step_samples: list[StepSample] = []
        self.total_preemptions: int = 0

    # 生命周期埋点
    def record_finished(self, seq: Sequence) -> None: ...
    def record_preemption(self, seq: Sequence, timestamp: float) -> None: ...
    def record_step(
        self,
        t_start: float,
        t_end: float,
        seqs: list[Sequence],
        is_prefill: bool,
        block_manager: BlockManager,
    ) -> None: ...

    # 查询
    def get_request_metrics(self, seq_id: int) -> RequestMetrics | None: ...
    def build(self) -> EngineMetrics: ...
```

## 5. 埋点位置

所有时间戳用 `time.perf_counter()`，仅在 rank 0 引擎进程采集。

### 5.1 `Sequence.__init__` 新增字段

```python
self.arrival_time: float | None = None
self.first_scheduled_time: float | None = None
self.first_token_time: float | None = None
self.finish_time: float | None = None
self.token_times: list[float] = []
self.preemption_count: int = 0
```

**重要**：这些字段**不进 `__getstate__`**——TP worker 不需要，避免 pickle IPC 变胖。

### 5.2 `Sequence.as_request_metrics()`

```python
def as_request_metrics(self) -> RequestMetrics:
    return RequestMetrics(
        arrival_time=self.arrival_time,
        first_scheduled_time=self.first_scheduled_time,
        first_token_time=self.first_token_time,
        finish_time=self.finish_time,
        token_times=list(self.token_times),
        num_prompt_tokens=self.num_prompt_tokens,
        num_completion_tokens=self.num_completion_tokens,
        preemption_count=self.preemption_count,
    )
```

### 5.3 `LLMEngine.add_request`

```python
seq = Sequence(prompt, sampling_params)
seq.arrival_time = perf_counter()
self.scheduler.add(seq)
```

### 5.4 `Scheduler.schedule`（prefill 分支）

在设置 `seq.num_scheduled_tokens = min(num_tokens, remaining)` 之前加：

```python
if seq.first_scheduled_time is None:
    seq.first_scheduled_time = perf_counter()
```

即使首次被调度的是一个 chunked prefill 的首 chunk，也视为首次调度。

### 5.5 `Scheduler.postprocess`

对每个生成 token 的事件打点：

```python
now = perf_counter()
seq.append_token(token_id)
seq.token_times.append(now)
if seq.first_token_time is None:
    seq.first_token_time = now
seq.num_cached_tokens += 1
seq.num_scheduled_tokens = 0
if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
    seq.status = SequenceStatus.FINISHED
    seq.finish_time = now
    self.block_manager.deallocate(seq)
    self.running.remove(seq)
    self.metrics.record_finished(seq)
```

注意：prefill 阶段只有最后一个 chunk 才会走到 `append_token`；前面的 chunk 走 `continue` 分支，不碰时间戳。

### 5.6 `Scheduler.preempt`

```python
def preempt(self, seq: Sequence):
    seq.status = SequenceStatus.WAITING
    seq.preemption_count += 1
    self.block_manager.deallocate(seq)
    self.waiting.appendleft(seq)
    self.metrics.record_preemption(seq, perf_counter())
```

### 5.7 `LLMEngine.step`

```python
def step(self):
    t0 = perf_counter()
    seqs, is_prefill = self.scheduler.schedule()
    num_tokens = sum(seq.num_scheduled_tokens for seq in seqs) if is_prefill else -len(seqs)
    token_ids = self.model_runner.call("run", seqs, is_prefill)
    self.scheduler.postprocess(seqs, token_ids, is_prefill)
    t1 = perf_counter()
    self.metrics.record_step(t0, t1, seqs, is_prefill, self.scheduler.block_manager)
    outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
    return outputs, num_tokens
```

### 5.8 侵入性评估

- `sequence.py`：+6 字段 + 1 方法（~15 行）
- `scheduler.py`：构造函数接 collector + 4 处埋点（~10 行）
- `llm_engine.py`：step 外包、reset_metrics、get_aggregate_metrics、add_request 打时间戳（~20 行）
- **完全不碰** `model_runner.py` / `block_manager.py` / `layers/**` / `models/**`
- **完全不碰** torch.compile / CUDA graph 代码路径

## 6. API 与对接

### 6.1 `LLMEngine`

```python
class LLMEngine:
    def __init__(self, model, **kwargs):
        ...
        self.metrics = MetricsCollector()
        self.scheduler = Scheduler(config, self.metrics)

    def get_aggregate_metrics(self) -> EngineMetrics:
        return self.metrics.build()

    def reset_metrics(self):
        self.metrics = MetricsCollector()
        self.scheduler.metrics = self.metrics
```

### 6.2 `LLM.generate()` 返回值扩展

```python
outputs = [
    {
        "text": self.tokenizer.decode(token_ids),
        "token_ids": token_ids,
        "metrics": self.metrics.get_request_metrics(seq_id),
    }
    for seq_id, token_ids in sorted(outputs_dict.items())
]
```

向后兼容：`output["text"]`、`output["token_ids"]` 行为不变。

### 6.3 `bench.py` 升级

```python
llm.generate(["Benchmark: "], SamplingParams())
llm.reset_metrics()

t = time.time()
llm.generate(prompt_token_ids, sampling_params, use_tqdm=False)
t = time.time() - t

metrics = llm.get_aggregate_metrics()
print(metrics.summary_table())

import json
with open("bench_metrics.json", "w") as f:
    json.dump(metrics.to_dict(), f, indent=2)
```

### 6.4 `summary_table()` 示例

```
=== Nano-vLLM Engine Metrics ===
Requests:           256 completed, 3 preempted
Wall clock:         12.34s
Prefill throughput: 25123 tok/s
Decode throughput:  12045 tok/s

Per-request latency (seconds):
                    avg     p50     p90     p95     p99     max
TTFT                0.12    0.10    0.22    0.28    0.45    0.52
TPOT (s/tok)        0.008   0.007   0.012   0.014   0.020   0.025
E2E                 3.20    3.10    5.20    5.80    7.10    8.30
Queue               0.05    0.02    0.12    0.18    0.32    0.41
Prefill             0.07    0.06    0.18    0.22    0.38    0.45
Decode              3.10    3.00    5.00    5.60    6.90    8.10

Inter-token latency (flat across all tokens, seconds):
avg 0.008   p50 0.007   p90 0.012   p99 0.020
```

### 6.5 `example.py` 演示

```python
for prompt, output in zip(prompts, outputs):
    m = output["metrics"]
    print(f"TTFT: {m.ttft*1000:.1f}ms, TPOT: {m.tpot*1000:.1f}ms/tok")
```

## 7. 边界情况

| 情况 | 处理 |
|---|---|
| chunked prefill 多 chunk | 只有最后一个 chunk 走 `append_token`；`first_token_time is None` 守卫确保只记一次 |
| 抢占并重跑 | `preemption_count += 1`；`first_scheduled_time` 与 `first_token_time` 不重置（语义是"第一次"）；`token_times` 自然接续 |
| `num_completion_tokens <= 1` | `tpot` 分母为 0，返回 `0.0` |
| `get_aggregate_metrics()` 无数据 | 返回 `total_requests=0` 的 `EngineMetrics`，所有 `Percentiles` 字段 `nan` |
| CUDA graph decode | `sampler(...).tolist()` 强制 GPU→CPU 同步；外层 `perf_counter()` 准确，不额外 `torch.cuda.synchronize()` |
| TP `world_size > 1` | 仅 rank 0 埋点；worker 的 `ModelRunner.loop()` 不变；KV cache 本来就只在 rank 0 |
| 时序数据内存 | `StepSample` ~50B/条，1000+ 条百 KB 量级，可接受 |

## 8. 验证

项目无测试套件，新增轻量 smoke test：`tests/test_metrics.py`

```python
def test_metrics_smoke():
    llm = LLM(MODEL_PATH, enforce_eager=True, max_model_len=1024)
    out = llm.generate(["hello world"] * 4,
                       SamplingParams(temperature=0.6, max_tokens=32))

    for o in out:
        m = o["metrics"]
        # 时间单调性
        assert m.arrival_time < m.first_scheduled_time <= m.first_token_time < m.finish_time
        # token_times 长度 = completion token 数
        assert len(m.token_times) == m.num_completion_tokens
        # ITL 全 > 0
        intervals = m.inter_token_intervals
        assert all(x > 0 for x in intervals) if intervals else True

    agg = llm.get_aggregate_metrics()
    assert agg.total_requests == 4
    assert agg.ttft.avg > 0
```

**交叉校验**：`bench.py` 外层 `time.time()` 算的吞吐应与 `EngineMetrics.decode_throughput` 接近（偏差来自 warmup 和边界），能当场检验埋点无漏。

## 9. 文件清单

**新建**

- `nanovllm/engine/metrics.py`（~200 行）
- `tests/test_metrics.py`（~50 行）

**修改**

- `nanovllm/engine/sequence.py`：+6 字段 + 1 方法
- `nanovllm/engine/scheduler.py`：构造函数接 collector + 4 处埋点
- `nanovllm/engine/llm_engine.py`：step 外包、reset/get API、add_request 打点
- `nanovllm/llm.py`：`generate()` 返回 dict 加 `metrics` 字段
- `bench.py`：打印结构化汇总 + JSON 导出
- `example.py`：+1 行演示

**明确不改**

- `nanovllm/engine/model_runner.py`
- `nanovllm/engine/block_manager.py`
- `nanovllm/layers/**` / `nanovllm/models/**` / `nanovllm/utils/**`

## 10. 开销分析

- per step：2 次 `perf_counter()` + 1 次 dict 构造 + 1 次 `block_manager.num_free_blocks` 属性读取 ≈ **微秒量级**
- per token append：1 次 `perf_counter()` + 1 次 list append ≈ **亚微秒**
- 对比：单 decode step 毫秒级，埋点开销 **< 0.1%**

## 11. 对后续项目的铺垫

本设计的产出将被后续三个子项目直接复用：

- **采样方式扩展**：扩展 `SamplingParams` 后，用 TTFT/TPOT 衡量不同采样（greedy / top-k / top-p）在 CUDA graph 路径下的延迟影响
- **chunked prefill 推广**：`StepSample.num_batched_tokens` 和 `num_seqs` 的时序能直接显示 batch 利用率是否提升
- **DP 并行**：`EngineMetrics` 可作为单副本指标，DP 协调层在此之上做跨副本聚合即可
