"""DPLLMEngine.get_aggregate_metrics 合并 N 份 worker snapshot 的单测。

不启动真实 worker，手工构造 snapshot 放进 fake pipe，
验证合并：step_samples 连接、completed_requests dict update、total_preemptions 求和。
"""
import threading

import torch.multiprocessing as mp

from nanovllm.engine.metrics import StepSample, RequestMetrics


class SnapshotEchoWorker:
    """收到 metrics_snapshot 就回一个预置 snapshot。"""
    def __init__(self, child_conn, snapshot: dict):
        self.child_conn = child_conn
        self.snapshot = snapshot
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        while True:
            try:
                msg = self.child_conn.recv()
            except EOFError:
                return
            if msg[0] == "metrics_snapshot":
                self.child_conn.send(self.snapshot)
            elif msg[0] == "exit":
                self.child_conn.send(None)
                return
            else:
                self.child_conn.send(None)


def _engine_with_snapshots(snapshots: list[dict]):
    from nanovllm.engine.dp_engine import DPLLMEngine
    engine = DPLLMEngine.__new__(DPLLMEngine)
    engine.data_parallel_size = len(snapshots)
    engine.next_dispatch_rank = 0
    engine.next_global_id = 0
    engine._exited = False
    engine.ps = []
    engine.pipes = []
    engine.tokenizer = None
    engine._workers = []
    for snap in snapshots:
        parent, child = mp.Pipe(duplex=True)
        engine.pipes.append(parent)
        engine._workers.append(SnapshotEchoWorker(child, snap))
    return engine


def _make_req_metrics(arrival: float, first_tok: float, finish: float, n_prompt=10, n_comp=5) -> RequestMetrics:
    return RequestMetrics(
        arrival_time=arrival,
        first_scheduled_time=arrival + 0.1,
        first_token_time=first_tok,
        finish_time=finish,
        token_times=[first_tok + i * 0.01 for i in range(n_comp)],
        num_prompt_tokens=n_prompt,
        num_completion_tokens=n_comp,
        preemption_count=0,
    )


def _make_step_sample(ts: float, is_prefill: bool, n_batched: int) -> StepSample:
    return StepSample(
        timestamp=ts,
        is_prefill=is_prefill,
        num_seqs=1,
        num_batched_tokens=n_batched,
        num_free_blocks=100,
        num_used_blocks=10,
        step_duration=0.01,
    )


class TestMetricsAggregation:
    def test_step_samples_concatenated(self):
        snap_a = {
            "step_samples": [_make_step_sample(1.0, True, 100), _make_step_sample(1.1, False, 1)],
            "completed_requests": {},
            "total_preemptions": 0,
        }
        snap_b = {
            "step_samples": [_make_step_sample(1.05, True, 200)],
            "completed_requests": {},
            "total_preemptions": 0,
        }
        engine = _engine_with_snapshots([snap_a, snap_b])
        em = engine.get_aggregate_metrics()
        assert len(em.step_samples) == 3

    def test_completed_requests_merged(self):
        snap_a = {
            "step_samples": [],
            "completed_requests": {0: _make_req_metrics(0.0, 1.0, 2.0), 2: _make_req_metrics(0.2, 1.2, 2.2)},
            "total_preemptions": 0,
        }
        snap_b = {
            "step_samples": [],
            "completed_requests": {1: _make_req_metrics(0.1, 1.1, 2.1)},
            "total_preemptions": 0,
        }
        engine = _engine_with_snapshots([snap_a, snap_b])
        em = engine.get_aggregate_metrics()
        assert em.total_requests == 3

    def test_total_preemptions_summed(self):
        snap_a = {"step_samples": [], "completed_requests": {}, "total_preemptions": 2}
        snap_b = {"step_samples": [], "completed_requests": {}, "total_preemptions": 5}
        snap_c = {"step_samples": [], "completed_requests": {}, "total_preemptions": 0}
        engine = _engine_with_snapshots([snap_a, snap_b, snap_c])
        em = engine.get_aggregate_metrics()
        assert em.total_preemptions == 7

    def test_empty_snapshots_no_crash(self):
        snap = {"step_samples": [], "completed_requests": {}, "total_preemptions": 0}
        engine = _engine_with_snapshots([snap, snap])
        em = engine.get_aggregate_metrics()
        assert em.total_requests == 0
        assert em.total_preemptions == 0
        assert em.step_samples == []
