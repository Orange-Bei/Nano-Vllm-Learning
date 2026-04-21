# Data Parallel (DP-only) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 新增 `DPLLMEngine`，支持 `LLM(model, data_parallel_size=N>1)` 启 N 个 worker 进程、各占 1 卡跑 `LLMEngine(TP=1)`，主进程做 round-robin 请求分发 + sync step 节拍 + 聚合 metrics。`data_parallel_size=1` 零退化走现有 `LLMEngine`。

**Architecture:** 多进程架构（技术强制：`model_runner.py:26` 的 `init_process_group` 即使 TP=1 也调用，同进程 2 个 LLMEngine 会 TCP 冲突）。主进程 `DPLLMEngine` 通过 `torch.mp.Pipe(duplex=True)` 与 N 个 worker 做同步 RPC；seq_id 主进程全局分配防跨 worker 冲突；metrics 聚合用临时 `MetricsCollector` 合并 N 份 snapshot。

**Tech Stack:** Python 3.10+、`torch.multiprocessing` (spawn)、pytest（已有）、现有 LLMEngine / Scheduler / MetricsCollector 不动。

**Spec reference:** `doc/specs/2026-04-21-data-parallel-design.md`

**Commit policy (IMPORTANT):** 本 plan **不做中间 commit**。每个 Task 结束只运行测试、验证通过即可。所有改动累积到 Task 6 最后一步，一次性整体 commit（spec + plan + 实现 + 测试）。本轮 spec 已按规则没单独 commit，所以不需要 `git reset --soft`。

---

## 文件结构

**新建**

- `nanovllm/engine/dp_engine.py` — `DPLLMEngine` 主类 + `_dp_worker_entry` worker 入口 + 消息 dispatch；单一职责=DP 编排层
- `tests/test_dp_dispatch.py` — round-robin + global seq_id + 消息收发（mock pipe，无 GPU）
- `tests/test_dp_metrics_aggregation.py` — snapshot 合并逻辑（无 GPU）
- `tests/test_dp_e2e.py` — 多 GPU 活性测试（独立 fixture，不走 conftest session LLM）

**修改**

- `nanovllm/config.py` — 加 `data_parallel_size: int = 1` 字段 + `__post_init__` 校验
- `nanovllm/engine/model_runner.py` — 1 行：`init_process_group` URL 从硬编码 `"tcp://localhost:2333"` 改为读 `NANO_VLLM_DIST_PORT` 环境变量（默认 2333）
- `nanovllm/llm.py` — `class LLM(LLMEngine): pass` 改为 `__new__` 工厂

**明确不改**

- `nanovllm/engine/llm_engine.py` / `scheduler.py` / `block_manager.py` / `sequence.py` / `metrics.py`
- `nanovllm/layers/**` / `nanovllm/models/**` / `nanovllm/utils/**` / `nanovllm/sampling_params.py`
- `tests/conftest.py`（session `llm` fixture 保留）
- `bench.py` / `example.py` / `pyproject.toml`

---

## Task 1: 前置改动（Config 校验 + model_runner 端口 env 化）

**Files:**
- Modify: `nanovllm/config.py`
- Modify: `nanovllm/engine/model_runner.py`

- [ ] **Step 1: Config 加字段 + 校验**

打开 `nanovllm/config.py`，加 `data_parallel_size: int = 1` 字段和 `__post_init__` 校验：

```python
import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass(slots=True)
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    data_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        assert self.data_parallel_size >= 1
        assert not (self.tensor_parallel_size > 1 and self.data_parallel_size > 1), \
            "DP × TP 暂不支持（本次 spec 非目标）"
        if self.data_parallel_size > 1:
            import torch
            assert self.data_parallel_size <= torch.cuda.device_count(), \
                f"data_parallel_size={self.data_parallel_size} > 可用 GPU 数 {torch.cuda.device_count()}"
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
```

要点：
- `data_parallel_size` 放在 `tensor_parallel_size` 之后（字段顺序一致）
- torch 导入放到 if 分支里，TP=1 默认场景零副作用（不触发 CUDA init）

- [ ] **Step 2: model_runner 端口 env 化**

打开 `nanovllm/engine/model_runner.py`，第 1 行加 `import os`（原来没有），然后改 `__init__` 里的 `init_process_group`：

原代码（line 26）：
```python
        dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)
```

改成：
```python
        _dist_port = os.environ.get("NANO_VLLM_DIST_PORT", "2333")
        dist.init_process_group("nccl", f"tcp://localhost:{_dist_port}", world_size=self.world_size, rank=rank)
```

- [ ] **Step 3: 跑现有测试回归**

```bash
cd /home/ubuntu/Nano-Vllm-Learning
pytest tests/ --ignore=tests/test_metrics_e2e.py --ignore=tests/test_sampling_e2e.py --ignore=tests/test_chunked_prefill_e2e.py -q
```

预期：全部通过（70 个单测）——Config 字段添加和 env var 改动不应影响任何现有行为。

**潜在坑**：若 `test_config.py` 或类似存在（目前没有），可能需要新增 `data_parallel_size` 断言。本 project 当前没有 Config 的单测文件，不需要处理。

---

## Task 2: DPLLMEngine 骨架 + 分发逻辑（TDD）

**Files:**
- Create: `nanovllm/engine/dp_engine.py`
- Create: `tests/test_dp_dispatch.py`

- [ ] **Step 1: 写 `tests/test_dp_dispatch.py` 的单测**

单测策略：不实际启动 worker 进程，用 `torch.multiprocessing.Pipe` 造"假 worker"——主进程 send 后，测试线程立刻 recv 并 echo ack。这样就能验证 DPLLMEngine 的 dispatch 逻辑、不需 GPU 不需 import torch.cuda。

创建 `tests/test_dp_dispatch.py`：

```python
"""DPLLMEngine 的请求分发单测。

不启动真实 worker 进程：用 torch.mp.Pipe 造 fake worker 线程，
验证主进程的 round-robin + global seq_id + 消息序列化。
"""
import threading
import pytest
import torch.multiprocessing as mp

from nanovllm.sampling_params import SamplingParams


class FakeWorker:
    """在独立线程里跑，接收主进程消息、echo ack 或 fake 响应。"""
    def __init__(self, child_conn):
        self.child_conn = child_conn
        self.received = []
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        while True:
            try:
                msg = self.child_conn.recv()
            except EOFError:
                return
            self.received.append(msg)
            method = msg[0]
            if method == "exit":
                self.child_conn.send(None)
                return
            elif method == "add":
                self.child_conn.send(None)
            elif method == "is_finished":
                self.child_conn.send(True)
            elif method == "metrics_snapshot":
                self.child_conn.send({"step_samples": [], "completed_requests": {}, "total_preemptions": 0})
            elif method == "reset_metrics":
                self.child_conn.send(None)
            else:
                self.child_conn.send(None)


def _make_engine_with_fakes(n: int):
    """构造 DPLLMEngine，但 pipe/ps 用 fake worker 替换掉。"""
    from nanovllm.engine.dp_engine import DPLLMEngine
    # 绕过 __init__，直接手工组装字段
    engine = DPLLMEngine.__new__(DPLLMEngine)
    engine.data_parallel_size = n
    engine.next_dispatch_rank = 0
    engine.next_global_id = 0
    engine._exited = False
    engine.ps = []
    engine.pipes = []
    engine.tokenizer = None  # 不 tokenize，只传 token_ids
    engine._fake_workers = []
    for _ in range(n):
        parent_conn, child_conn = mp.Pipe(duplex=True)
        engine.pipes.append(parent_conn)
        engine._fake_workers.append(FakeWorker(child_conn))
    return engine


class TestDispatch:
    def test_round_robin_3_workers(self):
        engine = _make_engine_with_fakes(3)
        sp = SamplingParams()
        gids = [engine.add_request([1, 2, 3], sp) for _ in range(6)]
        # global id 单调递增
        assert gids == [0, 1, 2, 3, 4, 5]
        # round-robin 分发：rank 0/1/2/0/1/2
        msgs = [(i, w.received) for i, w in enumerate(engine._fake_workers)]
        assert len(engine._fake_workers[0].received) == 2
        assert len(engine._fake_workers[1].received) == 2
        assert len(engine._fake_workers[2].received) == 2
        # 每条消息第一个 field 是 "add"
        for w in engine._fake_workers:
            for msg in w.received:
                assert msg[0] == "add"

    def test_message_format(self):
        engine = _make_engine_with_fakes(2)
        sp = SamplingParams(temperature=0.5, max_tokens=16)
        engine.add_request([10, 20, 30], sp)
        # worker 0 收到 ("add", gid=0, token_ids=[10,20,30], sp)
        msg = engine._fake_workers[0].received[0]
        assert msg[0] == "add"
        assert msg[1] == 0  # gid
        assert msg[2] == [10, 20, 30]
        assert msg[3].temperature == 0.5
        assert msg[3].max_tokens == 16

    def test_is_finished_all_true(self):
        engine = _make_engine_with_fakes(3)
        # FakeWorker 默认 is_finished → True
        assert engine.is_finished() is True

    def test_reset_metrics_broadcasts(self):
        engine = _make_engine_with_fakes(3)
        engine.reset_metrics()
        for w in engine._fake_workers:
            assert any(msg[0] == "reset_metrics" for msg in w.received)

    def test_exit_broadcasts(self):
        engine = _make_engine_with_fakes(2)
        engine.exit()
        for w in engine._fake_workers:
            assert any(msg[0] == "exit" for msg in w.received)
        # exit 幂等
        engine.exit()  # 不应抛
```

- [ ] **Step 2: 运行测试，确认 import 失败**

```bash
pytest tests/test_dp_dispatch.py -v
```

预期：`ImportError: cannot import name 'DPLLMEngine' from 'nanovllm.engine.dp_engine'`（文件还没建）。

- [ ] **Step 3: 创建 `nanovllm/engine/dp_engine.py`**

```python
"""Data Parallel 引擎。

N 个 worker 子进程，每进程独占 1 卡跑 LLMEngine(TP=1)。
主进程做 round-robin 请求分发 + sync step 节拍 + 聚合 metrics。
"""
import atexit
import os
from dataclasses import fields
from time import perf_counter

import torch.multiprocessing as mp
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from nanovllm.config import Config
from nanovllm.engine.metrics import EngineMetrics, MetricsCollector


def _dp_worker_entry(rank: int, config_kwargs: dict, child_conn):
    """DP worker 进程入口。必须先设 env var 再 import torch 相关模块。"""
    # 环境变量要在 import torch 之前——spawn context 保证子进程从干净 Python 起
    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
    os.environ["NANO_VLLM_DIST_PORT"] = str(2333 + rank)

    from time import perf_counter as _pc
    from nanovllm.engine.llm_engine import LLMEngine
    from nanovllm.engine.sequence import Sequence

    # worker 内强制 TP=1 / DP=1，防止递归
    kwargs = {**config_kwargs, "tensor_parallel_size": 1, "data_parallel_size": 1}
    engine = LLMEngine(**kwargs)

    try:
        while True:
            msg = child_conn.recv()
            method = msg[0]
            args = msg[1:]

            if method == "exit":
                child_conn.send(None)
                break

            elif method == "add":
                gid, token_ids, sp = args
                seq = Sequence(token_ids, sp)
                seq.seq_id = gid
                seq.arrival_time = _pc()
                engine.scheduler.add(seq)
                child_conn.send(None)

            elif method == "step":
                t0 = _pc()
                seqs, is_prefill = engine.scheduler.schedule()
                num_batched = sum(s.num_scheduled_tokens for s in seqs) if is_prefill else len(seqs)
                num_tokens = num_batched if is_prefill else -len(seqs)
                token_ids = engine.model_runner.call("run", seqs, is_prefill)
                engine.scheduler.postprocess(seqs, token_ids, is_prefill)
                t1 = _pc()
                engine.metrics.record_step(
                    t0, t1, seqs, is_prefill, engine.scheduler.block_manager, num_batched,
                )
                outputs = [(s.seq_id, s.completion_token_ids) for s in seqs if s.is_finished]
                child_conn.send((outputs, num_tokens, is_prefill))

            elif method == "is_finished":
                child_conn.send(engine.scheduler.is_finished())

            elif method == "metrics_snapshot":
                m = engine.metrics
                child_conn.send({
                    "step_samples": list(m.step_samples),
                    "completed_requests": dict(m.completed_requests),
                    "total_preemptions": m.total_preemptions,
                })

            elif method == "reset_metrics":
                engine.reset_metrics()
                child_conn.send(None)

            else:
                child_conn.send(("error", f"unknown method: {method}"))
    finally:
        engine.exit()


class DPLLMEngine:
    def __init__(self, model, **kwargs):
        config_field_names = {f.name for f in fields(Config)}
        config_kwargs_dict = {k: v for k, v in kwargs.items() if k in config_field_names}
        config_kwargs_dict["model"] = model
        config = Config(**config_kwargs_dict)
        self.data_parallel_size = config.data_parallel_size
        assert self.data_parallel_size > 1, "DPLLMEngine 仅用于 data_parallel_size > 1"

        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        config_kwargs_dict["eos"] = config.eos
        # hf_config 是 AutoConfig 对象，worker 会自己 load；不跨进程传
        config_kwargs_dict.pop("hf_config", None)

        ctx = mp.get_context("spawn")
        self.ps = []
        self.pipes = []
        for rank in range(self.data_parallel_size):
            parent_conn, child_conn = ctx.Pipe(duplex=True)
            p = ctx.Process(
                target=_dp_worker_entry,
                args=(rank, config_kwargs_dict, child_conn),
            )
            p.start()
            self.ps.append(p)
            self.pipes.append(parent_conn)

        self.next_dispatch_rank = 0
        self.next_global_id = 0
        self._exited = False
        atexit.register(self.exit)

    def add_request(self, prompt, sampling_params):
        if isinstance(prompt, str):
            token_ids = self.tokenizer.encode(prompt)
        else:
            token_ids = prompt
        gid = self.next_global_id
        self.next_global_id += 1
        rank = self.next_dispatch_rank
        self.next_dispatch_rank = (self.next_dispatch_rank + 1) % self.data_parallel_size
        self.pipes[rank].send(("add", gid, token_ids, sampling_params))
        self.pipes[rank].recv()  # ack
        return gid

    def is_finished(self):
        for conn in self.pipes:
            conn.send(("is_finished",))
        return all(conn.recv() for conn in self.pipes)

    def reset_metrics(self):
        for conn in self.pipes:
            conn.send(("reset_metrics",))
        for conn in self.pipes:
            conn.recv()

    def exit(self):
        if self._exited:
            return
        self._exited = True
        for conn in self.pipes:
            try:
                conn.send(("exit",))
                conn.recv()
            except (BrokenPipeError, EOFError):
                pass
        for p in self.ps:
            p.join(timeout=5)
```

注意：这一步先不实现 `step()` / `generate()` / `get_aggregate_metrics()`（留到 Task 3/4）。Task 2 单测只覆盖 add_request / is_finished / reset_metrics / exit。

- [ ] **Step 4: 跑单测，确认全部 PASS**

```bash
pytest tests/test_dp_dispatch.py -v
```

预期：5 个测试 PASS。

**潜在坑**：`Config(**kwargs)` 调 `__post_init__` 会 assert `os.path.isdir(model)`。单测里 `_make_engine_with_fakes` 绕过了 `__init__` 所以没触发——OK。

---

## Task 3: Metrics 聚合（TDD）

**Files:**
- Modify: `nanovllm/engine/dp_engine.py`
- Create: `tests/test_dp_metrics_aggregation.py`

- [ ] **Step 1: 写 `tests/test_dp_metrics_aggregation.py`**

创建 `tests/test_dp_metrics_aggregation.py`：

```python
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
        # 合并后应覆盖 3 条 request
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
```

- [ ] **Step 2: 跑测试，确认 fail（方法未实现）**

```bash
pytest tests/test_dp_metrics_aggregation.py -v
```

预期：`AttributeError: 'DPLLMEngine' object has no attribute 'get_aggregate_metrics'`

- [ ] **Step 3: 在 `dp_engine.py` 的 `DPLLMEngine` 里加 `get_aggregate_metrics` 方法**

在 `exit()` 方法之前插入：

```python
    def get_aggregate_metrics(self) -> EngineMetrics:
        merged = MetricsCollector()
        for conn in self.pipes:
            conn.send(("metrics_snapshot",))
        for conn in self.pipes:
            snap = conn.recv()
            merged.step_samples.extend(snap["step_samples"])
            merged.completed_requests.update(snap["completed_requests"])
            merged.total_preemptions += snap["total_preemptions"]
        return merged.build()
```

- [ ] **Step 4: 跑测试，确认 PASS**

```bash
pytest tests/test_dp_metrics_aggregation.py -v
```

预期：4 个测试 PASS。

- [ ] **Step 5: 跑完整 DP 单测回归**

```bash
pytest tests/test_dp_dispatch.py tests/test_dp_metrics_aggregation.py -v
```

预期：9 个测试全 PASS。

---

## Task 4: step / generate / LLM 门面工厂

**Files:**
- Modify: `nanovllm/engine/dp_engine.py`
- Modify: `nanovllm/llm.py`

> **说明**：step / generate 的正确性主要靠 Task 5 的 E2E 测试覆盖；这里只是把方法写齐、让 `generate()` 能跑。

- [ ] **Step 1: 在 `dp_engine.py` 的 `DPLLMEngine` 里加 `step` 方法**

在 `get_aggregate_metrics` 之前加：

```python
    def step(self):
        """Broadcast step 给所有 worker，聚合 outputs。"""
        for conn in self.pipes:
            conn.send(("step",))
        all_outputs = []
        total_num_tokens = 0
        for conn in self.pipes:
            outputs, num_tokens, _is_prefill = conn.recv()
            all_outputs.extend(outputs)
            total_num_tokens += num_tokens
        return all_outputs, total_num_tokens
```

**注意语义**：`num_tokens` 正负号表示 prefill/decode（LLMEngine.step 约定）。DP 下一个 tick 里不同 worker 可能是 prefill 或 decode——简化处理：直接求和（prefill 负值 worker + decode 正值 worker 会抵消，但实际运行时大多同步在同一阶段，影响不大）。这个字段主要用于 tqdm 显示，精度不关键。

- [ ] **Step 2: 在 `dp_engine.py` 加 `generate` 方法**

在 `step` 之后加：

```python
    def generate(self, prompts, sampling_params, use_tqdm=True):
        pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True, disable=not use_tqdm)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)

        gid_order = []
        for prompt, sp in zip(prompts, sampling_params):
            gid = self.add_request(prompt, sp)
            gid_order.append(gid)

        outputs = {}
        prefill_throughput = decode_throughput = 0.
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step()
            if num_tokens > 0:
                prefill_throughput = num_tokens / (perf_counter() - t)
            elif num_tokens < 0:
                decode_throughput = -num_tokens / (perf_counter() - t)
            pbar.set_postfix({
                "Prefill": f"{int(prefill_throughput)}tok/s",
                "Decode": f"{int(decode_throughput)}tok/s",
            })
            for gid, token_ids in output:
                outputs[gid] = token_ids
                pbar.update(1)
        pbar.close()

        # 一次拉取所有 worker 的 request metrics（避免多次 broadcast）
        for conn in self.pipes:
            conn.send(("metrics_snapshot",))
        merged_reqs = {}
        for conn in self.pipes:
            snap = conn.recv()
            merged_reqs.update(snap["completed_requests"])

        result = []
        for gid in gid_order:
            token_ids = outputs[gid]
            result.append({
                "text": self.tokenizer.decode(token_ids),
                "token_ids": token_ids,
                "metrics": merged_reqs.get(gid),
            })
        return result
```

- [ ] **Step 3: 改 `nanovllm/llm.py` 为工厂分派**

整体替换 `nanovllm/llm.py`：

```python
from nanovllm.engine.llm_engine import LLMEngine
from nanovllm.engine.dp_engine import DPLLMEngine


class LLM:
    """工厂门面：data_parallel_size>1 走 DPLLMEngine，否则 LLMEngine。

    注意：LLM 不再是类型（去掉了 LLM(LLMEngine) 继承关系）。
    """
    def __new__(cls, model, **kwargs):
        if kwargs.get("data_parallel_size", 1) > 1:
            return DPLLMEngine(model, **kwargs)
        return LLMEngine(model, **kwargs)
```

- [ ] **Step 4: 回归测试**

```bash
pytest tests/ --ignore=tests/test_metrics_e2e.py --ignore=tests/test_sampling_e2e.py --ignore=tests/test_chunked_prefill_e2e.py --ignore=tests/test_dp_e2e.py -q
```

预期：之前 70 个单测 + 本轮 9 个 DP 单测 = 79 个 PASS。

**潜在坑**：如果 `LLM(LLMEngine)` 继承被代码中某处依赖（例如 `isinstance(x, LLM)`），会有回归。已 grep 确认代码库没这种用法——但再跑一次 e2e 测试验证：

```bash
pytest tests/test_metrics_e2e.py tests/test_sampling_e2e.py tests/test_chunked_prefill_e2e.py -q
```

预期：14 个 e2e 测试 PASS（这些用 LLM(..., data_parallel_size 未传默认 1) 构造，走 LLMEngine 路径，完全不触发 DP 逻辑）。

---

## Task 5: E2E 活性测试（需 ≥2 GPU）

**Files:**
- Create: `tests/test_dp_e2e.py`

- [ ] **Step 1: 写 `tests/test_dp_e2e.py`**

创建 `tests/test_dp_e2e.py`：

```python
"""DP 端到端活性测试。需要 ≥2 GPU。

独立 module fixture：DP=2 的 LLM 实例；teardown 显式 exit() 释放所有 worker。
"""
import os

import pytest
import torch

MODEL_PATH = os.environ.get("NANO_VLLM_TEST_MODEL", "/data/models/Qwen3-0.6B")

pytestmark = pytest.mark.skipif(
    not os.path.isdir(MODEL_PATH) or torch.cuda.device_count() < 2,
    reason=f"需要模型 {MODEL_PATH} 和 ≥2 GPU",
)


@pytest.fixture(scope="module")
def dp_llm():
    from nanovllm import LLM
    instance = LLM(
        MODEL_PATH,
        enforce_eager=True,
        max_model_len=1024,
        tensor_parallel_size=1,
        data_parallel_size=2,
    )
    yield instance
    instance.exit()


def test_dp_generate_completes(dp_llm):
    """基础活性：16 条混合长度 prompt 都能完成。"""
    from nanovllm import SamplingParams

    prompts = [f"hello world {i}" for i in range(8)] + \
              [f"introduce yourself briefly (case {i})" for i in range(8)]
    sp = SamplingParams(temperature=0.0, max_tokens=8)

    outputs = dp_llm.generate(prompts, sp)

    assert len(outputs) == 16
    for o in outputs:
        assert o["text"], f"空输出: {o}"
        assert len(o["token_ids"]) > 0
        assert o["metrics"] is not None
        assert o["metrics"].ttft > 0


def test_dp_aggregate_metrics_has_multi_rank_steps(dp_llm):
    """聚合 metrics 应该能看到来自 2 个 worker 的 step（总数 > 单 worker 步数）。"""
    from nanovllm import SamplingParams

    dp_llm.reset_metrics()
    prompts = [f"hi {i}" for i in range(8)]
    sp = SamplingParams(temperature=0.0, max_tokens=4)
    dp_llm.generate(prompts, sp)

    agg = dp_llm.get_aggregate_metrics()
    # DP=2 下总 step_samples 应该 ≥ prefill + decode 覆盖 ≥ 2 步
    assert len(agg.step_samples) > 0
    assert agg.total_requests == 8


def test_dp_reset_metrics(dp_llm):
    """reset_metrics 后 aggregate metrics 清零。"""
    from nanovllm import SamplingParams

    # 先 generate 一次产生 metrics
    dp_llm.generate(["warm"], SamplingParams(temperature=0.0, max_tokens=2))
    agg_before = dp_llm.get_aggregate_metrics()
    assert agg_before.total_requests >= 1

    dp_llm.reset_metrics()
    agg_after = dp_llm.get_aggregate_metrics()
    assert agg_after.total_requests == 0
    assert agg_after.step_samples == []
```

- [ ] **Step 2: 跑 e2e 测试**

```bash
pytest tests/test_dp_e2e.py -v -s
```

预期：3 个测试 PASS。首次 ~30s（需要 load 2 份 model weights）。

**潜在坑：**

1. `torch.multiprocessing.Pipe` 的 spawn 兼容性：Python 3.10 下 spawn ctx 的 Pipe 要用 `ctx.Pipe(duplex=True)`，我们已经用了。
2. `CUDA_VISIBLE_DEVICES` 要在 worker 的 `_dp_worker_entry` 最开始、`import torch` 之前设——spawn 保证子进程是新 Python 解释器，所以这是成立的（entry function 内部 `import` 的 torch 是子进程第一次 import）。
3. 如果子进程 spawn 失败（比如模型路径在子进程里拿不到），主进程 recv 会 hang。加 `timeout` 诊断一下。
4. `pytest` 收集时也会 import 源文件，确保 `nanovllm/engine/dp_engine.py` 的 top-level 只做 import 不做耗时操作。

- [ ] **Step 3: 全套测试回归**

```bash
pytest tests/ -q
```

预期：70（round1+2+3）+ 9（DP 单测）+ 3（DP e2e）= 82 个 PASS。

---

## Task 6: bench sanity + 最终整体 commit

**Files:**
- 无文件新增/修改

- [ ] **Step 1: 跑 `bench.py` sanity check（DP=1 baseline）**

```bash
cd /home/ubuntu/Nano-Vllm-Learning
python bench.py 2>&1 | tail -15
```

预期：和之前跑过的数字量级相似（~5000+ tok/s 总吞吐）。bench.py 默认 `data_parallel_size=1`，走 LLMEngine，不应该受 DP 改动影响。

结果 JSON 会写到 `temp/bench-runs/bench_metrics.json`（Task 0d20e99 已改）。

- [ ] **Step 2: 跑 DP=2 bench（可选 sanity，需 2 GPU）**

临时修改 bench.py 的 `llm = LLM(path, ...)` 调用加 `data_parallel_size=2`，或者用一次性脚本：

```bash
python -c "
import time
from random import randint, seed
from nanovllm import LLM, SamplingParams

seed(0)
path = '/data/models/Qwen3-0.6B'
llm = LLM(path, enforce_eager=False, max_model_len=4096, data_parallel_size=2)
prompts = [[randint(0, 10000) for _ in range(randint(100, 1024))] for _ in range(256)]
sps = [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=randint(100, 1024)) for _ in range(256)]
llm.generate(['warmup'], SamplingParams())
llm.reset_metrics()
t = time.time()
llm.generate(prompts, sps, use_tqdm=False)
t = time.time() - t
total = sum(sp.max_tokens for sp in sps)
print(f'DP=2 Total: {total}tok, Time: {t:.2f}s, Throughput: {total/t:.2f}tok/s')
llm.exit()
"
```

预期：吞吐显著高于 DP=1（接近 1.8~2× 单卡）。如果 <1.5× 或报错，排查 IPC 瓶颈或 worker 配置。**非自动化断言，眼看即可。** 命令产出不 commit。

- [ ] **Step 3: 检查 git 状态**

```bash
git status
git log --oneline -4
```

预期：
- HEAD 在 `899d1f2`（上一个 roadmap 的 gitignore + bench.py 改动）
- Working tree 修改：`nanovllm/config.py`、`nanovllm/engine/model_runner.py`、`nanovllm/llm.py`
- Working tree 新增：`nanovllm/engine/dp_engine.py`、`tests/test_dp_dispatch.py`、`tests/test_dp_metrics_aggregation.py`、`tests/test_dp_e2e.py`、`doc/specs/2026-04-21-data-parallel-design.md`、`doc/plans/2026-04-21-data-parallel.md`

- [ ] **Step 4: 整体 commit**

```bash
git add \
  nanovllm/config.py \
  nanovllm/engine/model_runner.py \
  nanovllm/engine/dp_engine.py \
  nanovllm/llm.py \
  tests/test_dp_dispatch.py \
  tests/test_dp_metrics_aggregation.py \
  tests/test_dp_e2e.py \
  doc/specs/2026-04-21-data-parallel-design.md \
  doc/plans/2026-04-21-data-parallel.md

git commit -m "新增 DP 数据并行 (DP-only, TP=1 固定) 二次开发，含 spec/plan/DPLLMEngine/LLM 工厂/config+model_runner 小改/单测/端到端冒烟测试。"
```

**不加 `Co-Authored-By`**（memory 里的 commit 风格偏好）。

- [ ] **Step 5: 验证最终 git 状态**

```bash
git log --oneline -5
git show --stat HEAD | head -20
```

预期：
- 新 commit 在 HEAD
- 9 个文件改动（3 修改 + 6 新增）
- 上一个 commit 是 `899d1f2`

- [ ] **Step 6: 最后一次全套测试确认**

```bash
pytest tests/ 2>&1 | tail -3
```

预期：82 passed（round1/2/3/4 合计）。

---

## 完成判据

- [ ] `tests/test_dp_dispatch.py` 5 个单测 PASS
- [ ] `tests/test_dp_metrics_aggregation.py` 4 个单测 PASS
- [ ] `tests/test_dp_e2e.py` 3 个 e2e 测试 PASS（需 ≥2 GPU）
- [ ] round1/2/3 所有测试无回归
- [ ] `bench.py` DP=1 手工 sanity：吞吐不出现 >10% 退步
- [ ] DP=2 sanity（可选，需 2 GPU）：吞吐接近 1.8~2× 单卡
- [ ] 最终 git log：`899d1f2 ← 新 commit`，无中间 commit
