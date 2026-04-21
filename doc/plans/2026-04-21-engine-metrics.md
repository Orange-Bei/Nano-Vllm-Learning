# Engine Metrics (TTFT / TPOT / ITL / KV Utilization) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Nano-vLLM 引擎增加 per-request 与引擎级可观测性（TTFT / TPOT / E2E / queue / prefill / decode time、ITL 分布、KV cache 利用率、抢占计数、吞吐），为后续采样扩展、chunked prefill 推广、DP 并行三个子项目提供统一观测基础设施。

**Architecture:** 混合式——per-seq 时间戳挂在 `Sequence` 上（原生 primitive 字段），全局时序与事件数据收集到独立的 `MetricsCollector`；`get_aggregate_metrics()` 返回 `EngineMetrics` 对象，含 `p50/p90/p99` 分位数。

**Tech Stack:** Python 3.10+、pytest（新增 dev 依赖）、`time.perf_counter()`。不引入任何 GPU/异步/网络依赖。

**Spec reference:** `doc/specs/2026-04-21-engine-metrics-design.md`

---

## 文件结构

**新建**
- `nanovllm/engine/metrics.py` — 所有新数据结构与 collector
- `tests/__init__.py` — 让 `tests/` 成为包（空文件）
- `tests/test_metrics_types.py` — Task 1 的单测（Percentiles、RequestMetrics）
- `tests/test_metrics_collector.py` — Task 2 的单测（MetricsCollector、EngineMetrics）
- `tests/test_sequence_metrics.py` — Task 3 的单测（Sequence 时间戳、pickle、as_request_metrics）
- `tests/test_metrics_e2e.py` — Task 6 的端到端冒烟测试（需 GPU + 模型文件）

**修改**
- `nanovllm/engine/sequence.py` — 新增 6 个时间戳/计数字段 + `as_request_metrics()` 方法
- `nanovllm/engine/scheduler.py` — 构造函数新增 `metrics` 参数；`schedule` / `postprocess` / `preempt` 里共 4 处埋点
- `nanovllm/engine/llm_engine.py` — `__init__` 建 collector、`add_request` 打时间戳、`step` 外包采样、`generate` 返回 dict 加 `metrics` 字段、新增 `reset_metrics` / `get_aggregate_metrics`
- `bench.py` — 调用 `get_aggregate_metrics()` 打印汇总表 + JSON 导出
- `example.py` — 演示单请求 TTFT / TPOT 打印
- `pyproject.toml` — 增加 `[project.optional-dependencies]` 放 pytest

**明确不改**
- `nanovllm/engine/block_manager.py`（通过 `len(bm.free_block_ids)` / `len(bm.used_block_ids)` 访问）
- `nanovllm/engine/model_runner.py` / `nanovllm/layers/**` / `nanovllm/models/**` / `nanovllm/utils/**`
- 不触碰 torch.compile / CUDA graph 代码路径

---

## Task 1: 创建 `metrics.py` 的数据类型（Percentiles / StepSample / RequestMetrics）

**Files:**
- Modify: `pyproject.toml`
- Create: `nanovllm/engine/metrics.py`
- Create: `tests/__init__.py`
- Create: `tests/test_metrics_types.py`

- [ ] **Step 1: 安装 pytest 并在 pyproject 里登记**

编辑 `pyproject.toml`，在 `[project]` 的 `dependencies` 块之后添加：

```toml
[project.optional-dependencies]
dev = ["pytest>=8.0"]
```

然后安装：

```bash
pip install pytest
```

预期：`pytest --version` 能输出版本号。

- [ ] **Step 2: 创建 `tests/__init__.py`（空文件）**

```bash
touch tests/__init__.py
```

- [ ] **Step 3: 写 Percentiles 和 RequestMetrics 的失败单测**

创建 `tests/test_metrics_types.py`：

```python
import math
import pytest

from nanovllm.engine.metrics import Percentiles, RequestMetrics, StepSample


class TestPercentiles:
    def test_empty_samples_returns_nan(self):
        p = Percentiles.from_samples([])
        assert math.isnan(p.avg)
        assert math.isnan(p.min)
        assert math.isnan(p.p50)
        assert math.isnan(p.p90)
        assert math.isnan(p.p95)
        assert math.isnan(p.p99)
        assert math.isnan(p.max)

    def test_single_sample(self):
        p = Percentiles.from_samples([1.5])
        assert p.avg == 1.5
        assert p.min == 1.5
        assert p.p50 == 1.5
        assert p.p99 == 1.5
        assert p.max == 1.5

    def test_multiple_samples(self):
        p = Percentiles.from_samples([1.0, 2.0, 3.0, 4.0, 5.0])
        assert p.avg == 3.0
        assert p.min == 1.0
        assert p.max == 5.0
        assert p.p50 == 3.0


class TestRequestMetrics:
    def _make(self, arrival=0.0, first_sched=0.01, first_tok=0.1, finish=1.0,
              token_times=None, n_prompt=10, n_completion=5, preempt=0):
        if token_times is None:
            token_times = [0.1, 0.3, 0.5, 0.7, 1.0]
        return RequestMetrics(
            arrival_time=arrival,
            first_scheduled_time=first_sched,
            first_token_time=first_tok,
            finish_time=finish,
            token_times=token_times,
            num_prompt_tokens=n_prompt,
            num_completion_tokens=n_completion,
            preemption_count=preempt,
        )

    def test_ttft(self):
        m = self._make(arrival=0.0, first_tok=0.1)
        assert m.ttft == pytest.approx(0.1)

    def test_tpot_normal(self):
        m = self._make(first_tok=0.1, finish=1.0, n_completion=5)
        # (1.0 - 0.1) / (5 - 1) = 0.225
        assert m.tpot == pytest.approx(0.225)

    def test_tpot_single_completion_returns_zero(self):
        m = self._make(n_completion=1, finish=0.1)
        assert m.tpot == 0.0

    def test_tpot_zero_completion_returns_zero(self):
        m = self._make(n_completion=0, finish=0.1)
        assert m.tpot == 0.0

    def test_e2e_latency(self):
        m = self._make(arrival=0.0, finish=1.0)
        assert m.e2e_latency == pytest.approx(1.0)

    def test_queue_time(self):
        m = self._make(arrival=0.0, first_sched=0.05)
        assert m.queue_time == pytest.approx(0.05)

    def test_prefill_time(self):
        m = self._make(first_sched=0.05, first_tok=0.12)
        assert m.prefill_time == pytest.approx(0.07)

    def test_decode_time(self):
        m = self._make(first_tok=0.1, finish=1.0)
        assert m.decode_time == pytest.approx(0.9)

    def test_inter_token_intervals(self):
        m = self._make(token_times=[0.1, 0.3, 0.5])
        assert m.inter_token_intervals == pytest.approx([0.2, 0.2])

    def test_inter_token_intervals_single_token(self):
        m = self._make(token_times=[0.1])
        assert m.inter_token_intervals == []


class TestStepSample:
    def test_fields(self):
        s = StepSample(
            timestamp=0.0,
            is_prefill=True,
            num_seqs=4,
            num_batched_tokens=1024,
            num_free_blocks=100,
            num_used_blocks=20,
            step_duration=0.005,
        )
        assert s.num_seqs == 4
        assert s.is_prefill is True
```

- [ ] **Step 4: 运行测试确认失败**

```bash
cd /home/ubuntu/Nano-Vllm-Learning
python -m pytest tests/test_metrics_types.py -v
```

预期：`ModuleNotFoundError: No module named 'nanovllm.engine.metrics'`

- [ ] **Step 5: 写 `metrics.py` 的三个数据类**

创建 `nanovllm/engine/metrics.py`：

```python
"""引擎与请求级性能指标采集。

per-seq 时间戳挂在 Sequence 上，全局时序与事件数据在 MetricsCollector。
get_aggregate_metrics() 返回 EngineMetrics，包含 p50/p90/p99 分位数。
"""
import math
from dataclasses import dataclass, field


def _percentile(sorted_samples: list[float], pct: float) -> float:
    """线性插值分位数，pct in [0, 100]。"""
    if not sorted_samples:
        return float("nan")
    if len(sorted_samples) == 1:
        return sorted_samples[0]
    k = (len(sorted_samples) - 1) * (pct / 100.0)
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return sorted_samples[int(k)]
    return sorted_samples[lo] + (sorted_samples[hi] - sorted_samples[lo]) * (k - lo)


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
    def from_samples(cls, samples: list[float]) -> "Percentiles":
        if not samples:
            nan = float("nan")
            return cls(nan, nan, nan, nan, nan, nan, nan)
        s = sorted(samples)
        return cls(
            avg=sum(s) / len(s),
            min=s[0],
            p50=_percentile(s, 50),
            p90=_percentile(s, 90),
            p95=_percentile(s, 95),
            p99=_percentile(s, 99),
            max=s[-1],
        )


@dataclass
class StepSample:
    timestamp: float
    is_prefill: bool
    num_seqs: int
    num_batched_tokens: int
    num_free_blocks: int
    num_used_blocks: int
    step_duration: float


@dataclass
class RequestMetrics:
    arrival_time: float
    first_scheduled_time: float
    first_token_time: float
    finish_time: float
    token_times: list[float]

    num_prompt_tokens: int
    num_completion_tokens: int
    preemption_count: int

    @property
    def ttft(self) -> float:
        return self.first_token_time - self.arrival_time

    @property
    def tpot(self) -> float:
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

- [ ] **Step 6: 运行测试确认通过**

```bash
python -m pytest tests/test_metrics_types.py -v
```

预期：全部 14 个测试通过。

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml nanovllm/engine/metrics.py tests/__init__.py tests/test_metrics_types.py
git commit -m "$(cat <<'EOF'
新增 metrics.py 的数据类型层 (Percentiles / StepSample / RequestMetrics) 及单测，实现 TTFT/TPOT/queue/prefill/decode/ITL 派生指标的无依赖计算逻辑；登记 pytest 为 dev 可选依赖。

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: 在 `metrics.py` 里加 `MetricsCollector` 和 `EngineMetrics`

**Files:**
- Modify: `nanovllm/engine/metrics.py`
- Create: `tests/test_metrics_collector.py`

- [ ] **Step 1: 写 MetricsCollector / EngineMetrics 的失败单测**

创建 `tests/test_metrics_collector.py`：

```python
import pytest

from nanovllm.engine.metrics import (
    EngineMetrics,
    MetricsCollector,
    Percentiles,
    RequestMetrics,
    StepSample,
)


class FakeBlockManager:
    def __init__(self, free: int, used: int):
        self.free_block_ids = list(range(free))
        self.used_block_ids = set(range(used))


class FakeSeq:
    """只实现 MetricsCollector 需要读取的属性。"""
    def __init__(self, seq_id, arrival, first_sched, first_tok, finish,
                 token_times, n_prompt, n_completion, preempt):
        self.seq_id = seq_id
        self.arrival_time = arrival
        self.first_scheduled_time = first_sched
        self.first_token_time = first_tok
        self.finish_time = finish
        self.token_times = token_times
        self.num_prompt_tokens = n_prompt
        self._n_completion = n_completion
        self.preemption_count = preempt

    @property
    def num_completion_tokens(self):
        return self._n_completion

    def as_request_metrics(self):
        return RequestMetrics(
            arrival_time=self.arrival_time,
            first_scheduled_time=self.first_scheduled_time,
            first_token_time=self.first_token_time,
            finish_time=self.finish_time,
            token_times=list(self.token_times),
            num_prompt_tokens=self.num_prompt_tokens,
            num_completion_tokens=self._n_completion,
            preemption_count=self.preemption_count,
        )


def make_seq(seq_id=1, arrival=0.0, n_completion=5):
    return FakeSeq(
        seq_id=seq_id,
        arrival=arrival,
        first_sched=arrival + 0.01,
        first_tok=arrival + 0.1,
        finish=arrival + 1.0,
        token_times=[arrival + 0.1 * i for i in range(1, n_completion + 1)],
        n_prompt=10,
        n_completion=n_completion,
        preempt=0,
    )


class TestMetricsCollector:
    def test_record_finished_stores_request_metrics(self):
        c = MetricsCollector()
        s = make_seq(seq_id=1)
        c.record_finished(s)
        assert c.get_request_metrics(1) is not None
        assert c.get_request_metrics(1).num_completion_tokens == 5

    def test_record_preemption_increments_total(self):
        c = MetricsCollector()
        s = make_seq(seq_id=1)
        c.record_preemption(s, 0.5)
        c.record_preemption(s, 0.8)
        assert c.total_preemptions == 2

    def test_record_step_appends_sample(self):
        c = MetricsCollector()
        bm = FakeBlockManager(free=100, used=20)
        s = make_seq()
        c.record_step(0.0, 0.005, [s], is_prefill=True, block_manager=bm)
        assert len(c.step_samples) == 1
        sample = c.step_samples[0]
        assert sample.is_prefill is True
        assert sample.num_seqs == 1
        assert sample.num_free_blocks == 100
        assert sample.num_used_blocks == 20
        assert sample.step_duration == pytest.approx(0.005)


class TestEngineMetricsBuild:
    def test_empty_collector(self):
        c = MetricsCollector()
        em = c.build()
        assert em.total_requests == 0
        assert em.total_preemptions == 0
        import math
        assert math.isnan(em.ttft.avg)

    def test_single_request(self):
        c = MetricsCollector()
        s = make_seq(seq_id=1, n_completion=5)
        c.record_finished(s)
        em = c.build()
        assert em.total_requests == 1
        assert em.ttft.avg == pytest.approx(0.1)

    def test_throughput_from_steps(self):
        c = MetricsCollector()
        bm = FakeBlockManager(free=100, used=0)
        # 一个 prefill step: 1024 tokens, 0.05s
        c.record_step(0.0, 0.05, [make_seq()], is_prefill=True, block_manager=bm)
        c.step_samples[-1].num_batched_tokens = 1024
        # 一个 decode step: 4 seqs, 0.01s -> 4 tok / 0.01s = 400 tok/s
        c.record_step(0.06, 0.07, [make_seq(), make_seq(seq_id=2),
                                   make_seq(seq_id=3), make_seq(seq_id=4)],
                      is_prefill=False, block_manager=bm)
        # finish 一个请求以确保 total_requests > 0
        c.record_finished(make_seq(seq_id=1))
        em = c.build()
        assert em.prefill_throughput == pytest.approx(1024 / 0.05, rel=0.01)
        assert em.decode_throughput == pytest.approx(4 / 0.01, rel=0.01)

    def test_to_dict_is_json_serializable(self):
        import json
        c = MetricsCollector()
        c.record_finished(make_seq(seq_id=1))
        em = c.build()
        d = em.to_dict()
        # 能被 json.dumps 序列化（不抛异常）
        json.dumps(d)

    def test_summary_table_returns_string(self):
        c = MetricsCollector()
        c.record_finished(make_seq(seq_id=1))
        em = c.build()
        table = em.summary_table()
        assert isinstance(table, str)
        assert "TTFT" in table
        assert "TPOT" in table
```

- [ ] **Step 2: 运行测试确认失败**

```bash
python -m pytest tests/test_metrics_collector.py -v
```

预期：`ImportError: cannot import name 'MetricsCollector'`

- [ ] **Step 3: 往 `metrics.py` 追加 MetricsCollector 和 EngineMetrics**

在 `nanovllm/engine/metrics.py` 末尾追加：

```python
@dataclass
class EngineMetrics:
    ttft: Percentiles
    tpot: Percentiles
    e2e_latency: Percentiles
    queue_time: Percentiles
    prefill_time: Percentiles
    decode_time: Percentiles
    itl: Percentiles

    total_requests: int
    total_preemptions: int
    prefill_throughput: float
    decode_throughput: float
    wall_clock_seconds: float

    step_samples: list[StepSample] = field(default_factory=list)

    def to_dict(self) -> dict:
        def p_dict(p: Percentiles) -> dict:
            return {
                "avg": p.avg, "min": p.min, "max": p.max,
                "p50": p.p50, "p90": p.p90, "p95": p.p95, "p99": p.p99,
            }

        return {
            "total_requests": self.total_requests,
            "total_preemptions": self.total_preemptions,
            "prefill_throughput": self.prefill_throughput,
            "decode_throughput": self.decode_throughput,
            "wall_clock_seconds": self.wall_clock_seconds,
            "ttft": p_dict(self.ttft),
            "tpot": p_dict(self.tpot),
            "e2e_latency": p_dict(self.e2e_latency),
            "queue_time": p_dict(self.queue_time),
            "prefill_time": p_dict(self.prefill_time),
            "decode_time": p_dict(self.decode_time),
            "itl": p_dict(self.itl),
            "step_samples": [
                {
                    "timestamp": s.timestamp,
                    "is_prefill": s.is_prefill,
                    "num_seqs": s.num_seqs,
                    "num_batched_tokens": s.num_batched_tokens,
                    "num_free_blocks": s.num_free_blocks,
                    "num_used_blocks": s.num_used_blocks,
                    "step_duration": s.step_duration,
                }
                for s in self.step_samples
            ],
        }

    def summary_table(self) -> str:
        def row(name: str, p: Percentiles, fmt: str = "{:.4f}") -> str:
            vals = [p.avg, p.p50, p.p90, p.p95, p.p99, p.max]
            return f"{name:<18}" + "  ".join(fmt.format(v) for v in vals)

        lines = [
            "=== Nano-vLLM Engine Metrics ===",
            f"Requests:           {self.total_requests} completed, "
            f"{self.total_preemptions} preempted",
            f"Wall clock:         {self.wall_clock_seconds:.2f}s",
            f"Prefill throughput: {self.prefill_throughput:.0f} tok/s",
            f"Decode throughput:  {self.decode_throughput:.0f} tok/s",
            "",
            "Per-request latency (seconds):",
            f"{'':<18}{'avg':>8}{'p50':>8}{'p90':>8}{'p95':>8}{'p99':>8}{'max':>8}",
            row("TTFT",         self.ttft),
            row("TPOT (s/tok)", self.tpot, "{:.5f}"),
            row("E2E",          self.e2e_latency),
            row("Queue",        self.queue_time),
            row("Prefill",      self.prefill_time),
            row("Decode",       self.decode_time),
            "",
            "Inter-token latency (flat across all tokens, seconds):",
            f"  avg {self.itl.avg:.5f}  p50 {self.itl.p50:.5f}  "
            f"p90 {self.itl.p90:.5f}  p99 {self.itl.p99:.5f}",
        ]
        return "\n".join(lines)


class MetricsCollector:
    def __init__(self) -> None:
        self.completed_requests: dict[int, RequestMetrics] = {}
        self.step_samples: list[StepSample] = []
        self.total_preemptions: int = 0

    def record_finished(self, seq) -> None:
        self.completed_requests[seq.seq_id] = seq.as_request_metrics()

    def record_preemption(self, seq, timestamp: float) -> None:
        self.total_preemptions += 1

    def record_step(
        self,
        t_start: float,
        t_end: float,
        seqs: list,
        is_prefill: bool,
        block_manager,
    ) -> None:
        num_batched = sum(getattr(s, "num_scheduled_tokens", 0) for s in seqs) \
            if is_prefill else len(seqs)
        self.step_samples.append(StepSample(
            timestamp=t_start,
            is_prefill=is_prefill,
            num_seqs=len(seqs),
            num_batched_tokens=num_batched,
            num_free_blocks=len(block_manager.free_block_ids),
            num_used_blocks=len(block_manager.used_block_ids),
            step_duration=t_end - t_start,
        ))

    def get_request_metrics(self, seq_id: int) -> RequestMetrics | None:
        return self.completed_requests.get(seq_id)

    def build(self) -> EngineMetrics:
        reqs = list(self.completed_requests.values())

        ttft = Percentiles.from_samples([r.ttft for r in reqs])
        tpot = Percentiles.from_samples([r.tpot for r in reqs])
        e2e = Percentiles.from_samples([r.e2e_latency for r in reqs])
        qt = Percentiles.from_samples([r.queue_time for r in reqs])
        pt = Percentiles.from_samples([r.prefill_time for r in reqs])
        dt = Percentiles.from_samples([r.decode_time for r in reqs])

        all_itls: list[float] = []
        for r in reqs:
            all_itls.extend(r.inter_token_intervals)
        itl = Percentiles.from_samples(all_itls)

        prefill_steps = [s for s in self.step_samples if s.is_prefill]
        decode_steps = [s for s in self.step_samples if not s.is_prefill]

        prefill_dur = sum(s.step_duration for s in prefill_steps)
        prefill_tok = sum(s.num_batched_tokens for s in prefill_steps)
        prefill_tput = prefill_tok / prefill_dur if prefill_dur > 0 else 0.0

        decode_dur = sum(s.step_duration for s in decode_steps)
        decode_tok = sum(s.num_seqs for s in decode_steps)
        decode_tput = decode_tok / decode_dur if decode_dur > 0 else 0.0

        if reqs:
            wall = max(r.finish_time for r in reqs) - min(r.arrival_time for r in reqs)
        else:
            wall = 0.0

        return EngineMetrics(
            ttft=ttft, tpot=tpot, e2e_latency=e2e,
            queue_time=qt, prefill_time=pt, decode_time=dt, itl=itl,
            total_requests=len(reqs),
            total_preemptions=self.total_preemptions,
            prefill_throughput=prefill_tput,
            decode_throughput=decode_tput,
            wall_clock_seconds=wall,
            step_samples=list(self.step_samples),
        )
```

- [ ] **Step 4: 运行测试确认通过**

```bash
python -m pytest tests/test_metrics_collector.py -v
```

预期：全部 7 个测试通过。

- [ ] **Step 5: Commit**

```bash
git add nanovllm/engine/metrics.py tests/test_metrics_collector.py
git commit -m "$(cat <<'EOF'
新增 MetricsCollector 与 EngineMetrics：collector 记录 finished/preemption/step 事件，build() 输出含 p50/p90/p99 分位数与吞吐的聚合对象；to_dict() 可 JSON 序列化，summary_table() 输出结构化汇总表。

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: 为 `Sequence` 增加时间戳字段与 `as_request_metrics()`

**Files:**
- Modify: `nanovllm/engine/sequence.py`
- Create: `tests/test_sequence_metrics.py`

- [ ] **Step 1: 写 Sequence 扩展的失败单测**

创建 `tests/test_sequence_metrics.py`：

```python
import pickle

from nanovllm.engine.metrics import RequestMetrics
from nanovllm.engine.sequence import Sequence
from nanovllm.sampling_params import SamplingParams


def test_new_fields_default_values():
    seq = Sequence([1, 2, 3], SamplingParams())
    assert seq.arrival_time is None
    assert seq.first_scheduled_time is None
    assert seq.first_token_time is None
    assert seq.finish_time is None
    assert seq.token_times == []
    assert seq.preemption_count == 0


def test_getstate_does_not_include_metrics_fields():
    """确保 TP worker 的 IPC 不会因此变胖。"""
    seq = Sequence([1, 2, 3], SamplingParams())
    seq.arrival_time = 12345.0
    seq.first_scheduled_time = 12346.0
    seq.token_times = [1.0, 2.0, 3.0, 4.0, 5.0]
    state = seq.__getstate__()
    # __getstate__ 返回 6 个字段的 tuple
    assert len(state) == 6
    # 不应该出现 metrics 值
    assert 12345.0 not in state
    assert 12346.0 not in state
    # token_times list 不应作为一个整体出现在 state 中
    for item in state:
        if isinstance(item, list):
            assert item != [1.0, 2.0, 3.0, 4.0, 5.0]


def test_pickle_roundtrip_preserves_core_state():
    """IPC 兼容性：pickle 来回走一遭，核心状态仍正确。"""
    seq = Sequence([1, 2, 3, 4, 5], SamplingParams(max_tokens=10))
    seq.arrival_time = 1.0  # 不会被 pickle
    seq.num_cached_tokens = 2
    seq.num_scheduled_tokens = 3
    seq.block_table = [10, 11]

    pkl = pickle.dumps(seq)
    restored = pickle.loads(pkl)

    assert restored.num_tokens == 5
    assert restored.num_prompt_tokens == 5
    assert restored.num_cached_tokens == 2
    assert restored.num_scheduled_tokens == 3
    assert restored.block_table == [10, 11]


def test_as_request_metrics():
    seq = Sequence([1, 2, 3, 4, 5], SamplingParams(max_tokens=3))
    seq.arrival_time = 0.0
    seq.first_scheduled_time = 0.01
    seq.first_token_time = 0.1
    seq.finish_time = 1.0
    seq.token_times = [0.1, 0.5, 1.0]
    seq.preemption_count = 1
    # 模拟已生成 3 个 completion token
    seq.append_token(100)
    seq.append_token(101)
    seq.append_token(102)

    m = seq.as_request_metrics()
    assert isinstance(m, RequestMetrics)
    assert m.arrival_time == 0.0
    assert m.first_token_time == 0.1
    assert m.token_times == [0.1, 0.5, 1.0]
    assert m.num_prompt_tokens == 5
    assert m.num_completion_tokens == 3
    assert m.preemption_count == 1
```

- [ ] **Step 2: 运行测试确认失败**

```bash
python -m pytest tests/test_sequence_metrics.py -v
```

预期：`AttributeError: 'Sequence' object has no attribute 'arrival_time'`

- [ ] **Step 3: 修改 `sequence.py`**

编辑 `nanovllm/engine/sequence.py`，在 `__init__` 末尾（当前第 30 行 `self.ignore_eos = ...` 之后）追加：

```python
        # --- metrics 字段（不参与 __getstate__，不进入 TP worker IPC）---
        self.arrival_time: float | None = None
        self.first_scheduled_time: float | None = None
        self.first_token_time: float | None = None
        self.finish_time: float | None = None
        self.token_times: list[float] = []
        self.preemption_count: int = 0
```

然后在文件末尾（`__setstate__` 之后）追加新方法：

```python
    def as_request_metrics(self):
        from nanovllm.engine.metrics import RequestMetrics
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

注意：局部 import 避免 metrics.py 和 sequence.py 形成循环 import（metrics.py 不能在顶层 import Sequence，因为 sequence.py 在 metrics.py 之前被 engine 模块加载）。

- [ ] **Step 4: 运行测试确认通过**

```bash
python -m pytest tests/test_sequence_metrics.py -v
```

预期：4 个测试全部通过。

- [ ] **Step 5: 再跑一遍前两个任务的单测，确保没回归**

```bash
python -m pytest tests/ -v
```

预期：所有任务的测试合计 25 个全部通过。

- [ ] **Step 6: Commit**

```bash
git add nanovllm/engine/sequence.py tests/test_sequence_metrics.py
git commit -m "$(cat <<'EOF'
Sequence 新增 6 个 metrics 字段 (arrival_time/first_scheduled_time/first_token_time/finish_time/token_times/preemption_count) 与 as_request_metrics() 方法；这些字段不进 __getstate__，不会增加 TP worker 的 IPC 体积。

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: `Scheduler` 埋点（构造函数 + 4 处）

**Files:**
- Modify: `nanovllm/engine/scheduler.py`

本任务没有独立单测，验证放在 Task 6 的 E2E。

- [ ] **Step 1: 修改 `Scheduler.__init__` 签名，接受 metrics collector**

编辑 `nanovllm/engine/scheduler.py`，顶部 import 块追加：

```python
from time import perf_counter

from nanovllm.engine.metrics import MetricsCollector
```

把 `class Scheduler:` 下的 `__init__` 改成：

```python
    def __init__(self, config: Config, metrics: MetricsCollector):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
        self.metrics = metrics
```

- [ ] **Step 2: 在 `schedule` 的 prefill 分支打"首次调度"时间戳**

找到 `scheduler.py` 里（原第 46 行）：

```python
            seq.num_scheduled_tokens = min(num_tokens, remaining)
```

把这行替换为：

```python
            if seq.first_scheduled_time is None:
                seq.first_scheduled_time = perf_counter()
            seq.num_scheduled_tokens = min(num_tokens, remaining)
```

- [ ] **Step 3: 在 `postprocess` 打 token_times / first_token_time / finish_time / record_finished**

把 `postprocess` 方法整个替换为：

```python
    def postprocess(self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool):
        for seq, token_id in zip(seqs, token_ids):
            if is_prefill:
                seq.num_cached_tokens = min(seq.num_cached_tokens + seq.num_scheduled_tokens, seq.num_tokens)
                if seq.num_cached_tokens < seq.num_tokens or seq.num_completion_tokens > 0:
                    seq.num_scheduled_tokens = 0
                    continue
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

- [ ] **Step 4: 在 `preempt` 累加抢占计数**

把 `preempt` 方法替换为：

```python
    def preempt(self, seq: Sequence):
        now = perf_counter()
        seq.status = SequenceStatus.WAITING
        seq.preemption_count += 1
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)
        self.metrics.record_preemption(seq, now)
```

- [ ] **Step 5: 快速语法检查（不跑测试）**

```bash
python -c "from nanovllm.engine.scheduler import Scheduler; print('ok')"
```

预期：输出 `ok`，无 `SyntaxError`/`ImportError`。

- [ ] **Step 6: 跑前三个任务的单测，确保没破坏**

```bash
python -m pytest tests/test_metrics_types.py tests/test_metrics_collector.py tests/test_sequence_metrics.py -v
```

预期：全部通过（共 25 条）。

- [ ] **Step 7: Commit**

```bash
git add nanovllm/engine/scheduler.py
git commit -m "$(cat <<'EOF'
Scheduler 接入 MetricsCollector：构造函数新增 metrics 参数；schedule 的 prefill 分支首次调度时打 first_scheduled_time；postprocess 每生成一个 token 写 token_times + first_token_time，FINISHED 时写 finish_time 并调 record_finished；preempt 累加 preemption_count 并调 record_preemption。

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: `LLMEngine` 接入 collector + 扩展 `generate()` 返回

**Files:**
- Modify: `nanovllm/engine/llm_engine.py`

本任务也不加独立单测，验证放在 Task 6。

- [ ] **Step 1: 修改 `llm_engine.py` 顶部 import**

编辑 `nanovllm/engine/llm_engine.py`，在现有 import 块末尾追加：

```python
from nanovllm.engine.metrics import EngineMetrics, MetricsCollector
```

- [ ] **Step 2: 在 `__init__` 里建 collector 并传给 Scheduler**

找到（当前 34 行附近）：

```python
        self.scheduler = Scheduler(config)
```

替换为：

```python
        self.metrics = MetricsCollector()
        self.scheduler = Scheduler(config, self.metrics)
```

- [ ] **Step 3: 在 `add_request` 打 arrival_time**

把 `add_request` 方法替换为：

```python
    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        seq.arrival_time = perf_counter()
        self.scheduler.add(seq)
```

注意：`perf_counter` 在文件顶部已经从 `time` import 过了。

- [ ] **Step 4: 在 `step` 外包 collector.record_step**

把 `step` 方法替换为：

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

- [ ] **Step 5: 新增 `get_aggregate_metrics` 与 `reset_metrics`**

在 `step` 方法之后、`is_finished` 之前插入：

```python
    def get_aggregate_metrics(self) -> EngineMetrics:
        return self.metrics.build()

    def reset_metrics(self):
        self.metrics = MetricsCollector()
        self.scheduler.metrics = self.metrics
```

- [ ] **Step 6: 修改 `generate` 的输出组装，加 metrics 字段**

找到 `generate` 方法末尾（当前 88-89 行）：

```python
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        return outputs
```

替换为：

```python
        sorted_ids = sorted(outputs.keys())
        result = []
        for seq_id in sorted_ids:
            token_ids = outputs[seq_id]
            result.append({
                "text": self.tokenizer.decode(token_ids),
                "token_ids": token_ids,
                "metrics": self.metrics.get_request_metrics(seq_id),
            })
        return result
```

- [ ] **Step 7: 语法与 import 检查**

```bash
python -c "from nanovllm.engine.llm_engine import LLMEngine; print('ok')"
```

预期：`ok`。

- [ ] **Step 8: 单测全跑一遍，确保无回归**

```bash
python -m pytest tests/ -v
```

预期：25 条全部通过。

- [ ] **Step 9: Commit**

```bash
git add nanovllm/engine/llm_engine.py
git commit -m "$(cat <<'EOF'
LLMEngine 接入 MetricsCollector：__init__ 建 collector 并注入 Scheduler；add_request 打 arrival_time；step 外包 record_step；新增 get_aggregate_metrics/reset_metrics；generate 返回的每个 dict 增加 metrics 字段（含该请求完整 RequestMetrics），向后兼容原有 text/token_ids 字段。

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: 端到端冒烟测试（需 GPU + Qwen3-0.6B 模型）

**Files:**
- Create: `tests/test_metrics_e2e.py`

- [ ] **Step 1: 写端到端冒烟测试**

创建 `tests/test_metrics_e2e.py`：

```python
"""端到端冒烟测试：需要 GPU 和 Qwen3-0.6B 模型。

如果模型路径不存在则自动跳过。
"""
import os
import pytest

MODEL_PATH = os.environ.get("NANO_VLLM_TEST_MODEL", "/data/models/Qwen3-0.6B")
pytestmark = pytest.mark.skipif(
    not os.path.isdir(MODEL_PATH),
    reason=f"Test model not found at {MODEL_PATH}",
)


@pytest.fixture(scope="module")
def llm():
    from nanovllm import LLM
    return LLM(MODEL_PATH, enforce_eager=True, max_model_len=1024,
               tensor_parallel_size=1)


def test_per_request_metrics_monotonicity(llm):
    from nanovllm import SamplingParams
    out = llm.generate(
        ["hello world", "introduce yourself"],
        SamplingParams(temperature=0.6, max_tokens=16),
    )
    assert len(out) == 2
    for o in out:
        m = o["metrics"]
        assert m is not None
        # 时间单调性
        assert m.arrival_time <= m.first_scheduled_time
        assert m.first_scheduled_time <= m.first_token_time
        assert m.first_token_time <= m.finish_time
        # token_times 长度 = completion token 数
        assert len(m.token_times) == m.num_completion_tokens
        # TTFT / E2E 都是正数
        assert m.ttft > 0
        assert m.e2e_latency > 0


def test_aggregate_metrics_nonempty(llm):
    from nanovllm import SamplingParams
    llm.reset_metrics()
    llm.generate(
        ["hello"] * 4,
        SamplingParams(temperature=0.6, max_tokens=16, ignore_eos=True),
    )
    agg = llm.get_aggregate_metrics()
    assert agg.total_requests == 4
    assert agg.ttft.avg > 0
    assert agg.decode_throughput > 0
    # 至少有 prefill 和 decode 两类 step 样本
    assert any(s.is_prefill for s in agg.step_samples)
    assert any(not s.is_prefill for s in agg.step_samples)


def test_to_dict_is_json_serializable(llm):
    import json
    from nanovllm import SamplingParams
    llm.reset_metrics()
    llm.generate(["hello"], SamplingParams(temperature=0.6, max_tokens=8, ignore_eos=True))
    agg = llm.get_aggregate_metrics()
    # 不抛异常即通过
    json.dumps(agg.to_dict())


def test_summary_table_contains_expected_rows(llm):
    from nanovllm import SamplingParams
    llm.reset_metrics()
    llm.generate(["hello"], SamplingParams(temperature=0.6, max_tokens=8, ignore_eos=True))
    table = llm.get_aggregate_metrics().summary_table()
    for key in ["TTFT", "TPOT", "E2E", "Queue", "Prefill", "Decode",
                "Inter-token latency"]:
        assert key in table, f"summary_table 缺少 '{key}' 行"


def test_reset_metrics_clears_state(llm):
    from nanovllm import SamplingParams
    llm.reset_metrics()
    llm.generate(["hello"], SamplingParams(temperature=0.6, max_tokens=4, ignore_eos=True))
    assert llm.get_aggregate_metrics().total_requests == 1
    llm.reset_metrics()
    assert llm.get_aggregate_metrics().total_requests == 0
```

- [ ] **Step 2: 运行端到端测试**

```bash
python -m pytest tests/test_metrics_e2e.py -v -s
```

预期：5 条测试通过；若模型路径不存在则显示 `SKIPPED`。

- [ ] **Step 3: 整套回归**

```bash
python -m pytest tests/ -v
```

预期：30 条全过（若 E2E 被 skip 则 25 条过 + 5 条 skip）。

- [ ] **Step 4: Commit**

```bash
git add tests/test_metrics_e2e.py
git commit -m "$(cat <<'EOF'
新增端到端冒烟测试 test_metrics_e2e.py，覆盖单请求时间戳单调性、聚合指标非空、to_dict JSON 可序列化、summary_table 行完整、reset_metrics 清空状态；若 Qwen3-0.6B 模型不在 /data/models 则自动 skip。

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: 更新 `bench.py` 打印汇总表 + JSON 导出

**Files:**
- Modify: `bench.py`

- [ ] **Step 1: 替换 `bench.py` 主体**

把整个 `bench.py` 覆盖为：

```python
import json
import os
import time
from random import randint, seed
from nanovllm import LLM, SamplingParams


def main():
    seed(0)
    num_seqs = 256
    max_input_len = 1024
    max_ouput_len = 1024

    path = os.path.expanduser("/data/models/Qwen3-0.6B")
    llm = LLM(path, enforce_eager=False, max_model_len=4096)

    prompt_token_ids = [[randint(0, 10000) for _ in range(randint(100, max_input_len))] for _ in range(num_seqs)]
    sampling_params = [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=randint(100, max_ouput_len)) for _ in range(num_seqs)]

    # warmup
    llm.generate(["Benchmark: "], SamplingParams())
    llm.reset_metrics()

    t = time.time()
    llm.generate(prompt_token_ids, sampling_params, use_tqdm=False)
    t = time.time() - t

    total_tokens = sum(sp.max_tokens for sp in sampling_params)
    throughput = total_tokens / t
    print(f"Total: {total_tokens}tok, Time: {t:.2f}s, Throughput: {throughput:.2f}tok/s")
    print()

    metrics = llm.get_aggregate_metrics()
    print(metrics.summary_table())

    json_path = "bench_metrics.json"
    with open(json_path, "w") as f:
        json.dump(metrics.to_dict(), f, indent=2)
    print(f"\nRaw metrics exported to {json_path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 本地跑一遍 bench.py（若有 GPU + 模型）**

```bash
python bench.py
```

预期：
- 原来的 `Total: ... Throughput: ...` 单行输出依然存在（向后兼容）
- 之后打印结构化 metrics 汇总表
- 当前目录下生成 `bench_metrics.json`

如果没有 GPU / 模型，跳过此步但仍执行 Step 3 的 commit。

- [ ] **Step 3: Commit**

```bash
git add bench.py
git commit -m "$(cat <<'EOF'
bench.py 升级：warmup 后调 reset_metrics，generate 完成后打印 summary_table 汇总表（TTFT/TPOT/E2E/Queue/Prefill/Decode 的 p50-p99、ITL 分布、吞吐、抢占数），并把完整 step_samples 时序导出到 bench_metrics.json；保留原有吞吐单行输出向后兼容。

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: 更新 `example.py` 演示 TTFT / TPOT

**Files:**
- Modify: `example.py`

- [ ] **Step 1: 在 example.py 的打印循环里加 metrics 行**

编辑 `example.py`，把尾部的打印循环：

```python
    for prompt, output in zip(prompts, outputs):
        print("\n")
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {output['text']!r}")
```

替换为：

```python
    for prompt, output in zip(prompts, outputs):
        print("\n")
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {output['text']!r}")
        m = output["metrics"]
        print(f"Metrics: TTFT={m.ttft*1000:.1f}ms, "
              f"TPOT={m.tpot*1000:.2f}ms/tok, "
              f"E2E={m.e2e_latency:.2f}s, "
              f"tokens={m.num_completion_tokens}")
```

- [ ] **Step 2: 本地跑一遍 example.py（若有 GPU + 模型）**

```bash
python example.py
```

预期：每条 prompt 下面多打一行 `Metrics: TTFT=... TPOT=... E2E=... tokens=...`。

- [ ] **Step 3: Commit**

```bash
git add example.py
git commit -m "$(cat <<'EOF'
example.py 演示 metrics 字段：在每条 completion 之后打印 TTFT/TPOT/E2E 延迟与生成 token 数，向用户展示二次开发后的可观测性输出。

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## 最终验证

- [ ] **Step 1: 整套测试回归**

```bash
python -m pytest tests/ -v
```

预期：30 条全过（或 25 过 + 5 skip 若无模型）。

- [ ] **Step 2: 确认 git 历史**

```bash
git log --oneline -10
```

预期：能看到 8 条本次项目的 commits。

- [ ] **Step 3: 手动核对 spec 覆盖**

对照 `doc/specs/2026-04-21-engine-metrics-design.md` 第 2.1 节目标清单：

- ✅ per-request 指标（arrival / first_sched / first_token / finish / token_times / preemption_count + 派生）
- ✅ 引擎级指标（KV 利用率时序、抢占事件、step 占满率、吞吐）
- ✅ API：generate 返回加 metrics 字段、get_aggregate_metrics、reset_metrics
- ✅ bench.py 打印汇总表

---

## 自检 Recap

**Spec coverage**：每条 spec 目标都对应至少一个 task，无遗漏。

**Placeholder scan**：所有 step 都含完整代码或命令；无 TBD/TODO/"similar to"。

**Type consistency**：
- `MetricsCollector.record_step(t_start, t_end, seqs, is_prefill, block_manager)` 在 Task 2 定义，Task 5 的 `llm_engine.step` 按此签名调用 ✅
- `Sequence.as_request_metrics()` 在 Task 3 定义，Task 2 的 `FakeSeq.as_request_metrics` 保持同名同返回类型 ✅
- `EngineMetrics` 的字段（`ttft`/`tpot`/`.../step_samples`）在 Task 2 定义，Task 6 的 E2E 测试按此读取 ✅
- `perf_counter` 统一从 `time` 模块 import（`scheduler.py` 在 Task 4 新增 import，`llm_engine.py` 中原本已有）✅
