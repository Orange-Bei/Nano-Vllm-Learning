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
        num_batched_tokens: int,
    ) -> None:
        self.step_samples.append(StepSample(
            timestamp=t_start,
            is_prefill=is_prefill,
            num_seqs=len(seqs),
            num_batched_tokens=num_batched_tokens,
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
