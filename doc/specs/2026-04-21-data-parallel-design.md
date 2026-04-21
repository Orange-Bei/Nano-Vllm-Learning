# Data Parallel（DP-only）· 设计文档

- **日期**：2026-04-21
- **作者**：Orange-Bei（与 Claude brainstorming）
- **状态**：Draft，待评审
- **Roadmap 项**：二次开发 #4

---

## 1. 背景

Nano-vLLM 目前只支持 TP（单副本多卡切 model 层）。3×4090 + Qwen3-0.6B（16 attention heads）场景下：

- **TP=3 不可行**：16 heads 不能整除 3
- **TP=2**：可行，但浪费一张卡
- **TP=1**：当前默认，只用 1 张卡

为把 3 卡的 compute 用满，需要 DP（Data Parallel）——每张卡放一份完整 model 副本，请求分配到不同副本上并行执行。DP 与 TP 的关键区别：副本之间**不做参数同步**（和 DDP 训练不同），每个副本独立跑 scheduler / forward，纯 per-request 级并行。

本改动是 roadmap 第 4 项、也是目前为止结构上最重的一次。

### 1.1 关键技术约束（已验证）

`nanovllm/engine/model_runner.py:26` 无条件调用：

```python
dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)
```

即使 `tensor_parallel_size=1` 时也会走这条路径。**同一 Python 进程里同时存在 2 个 LLMEngine 实例会冲突**（相同 TCP rendezvous URL、默认 process group 只能一份）。结论：**DP 必须走多进程架构**，每个 DP 副本独立 Python 进程。

## 2. 目标与非目标

### 2.1 目标

- 用户 API 新增 `data_parallel_size` 参数，默认 1；`data_parallel_size=1` 走现有 `LLMEngine`，**零退化**
- `data_parallel_size=N>1` 时启 N 个 worker 子进程，每进程独占一卡（`CUDA_VISIBLE_DEVICES=rank`），内部跑完整 `LLMEngine(tensor_parallel_size=1)`
- **Round-robin** 请求分发
- **Sync step 节拍**：主进程每 tick broadcast `step()`，等所有 worker 返回再进入下一 tick
- **对外 metrics 完全兼容**：`get_aggregate_metrics()` 返回聚合后的 `EngineMetrics`
- E2E：DP=2 和 DP=3 在 3×4090 上跑通，吞吐显著高于 DP=1

### 2.2 非目标（YAGNI）

- **不支持 DP × TP 组合**：`tensor_parallel_size>1 and data_parallel_size>1` assert 拒绝；这是下一项独立 scope
- **不做** least-loaded / random 分发策略
- **不做** async worker loops：节拍依赖主进程协调，接受"等最慢 worker"的极小开销
- **不做** worker 崩溃健康检查 / 自动重启：任何 worker 进程异常即主进程整体失败
- **不暴露** per-replica metrics：外部只看聚合后的全局视图
- **不共享** prefix cache：每 worker 独立 block hash；同 prompt 打到不同 worker 不复用 cache。这是 DP 本质成本
- **不做** 跨节点 DP：单机多 GPU 限定

## 3. 架构

### 3.1 进程拓扑

```
主进程 DPLLMEngine
  ├── tokenizer（仅此一份；主进程 tokenize 后给 worker 发 token_ids）
  ├── next_global_seq_id counter（全局 seq_id 分配）
  ├── next_dispatch_rank counter（round-robin）
  ├── pipes[rank] ↔ worker[rank] 进程（torch.mp.Pipe duplex）
  └── 对外 API: add_request / generate / get_aggregate_metrics /
                 reset_metrics / is_finished / exit

Workers（N 个子进程，rank = 0..N-1）
  ├── LLMEngine 实例（tensor_parallel_size=1, CUDA_VISIBLE_DEVICES=rank）
  ├── 消息 loop: 阻塞 recv → dispatch 到对应方法 → send 返回值
  └── Sequence.seq_id 被主进程分配的 global_id override
```

### 3.2 控制流（sync 节拍）

1. 用户调 `generate(prompts, sps)`；主进程 tokenize 后按 round-robin 把 `("add", gid, token_ids, sp)` 通过 pipe 发给选中 worker
2. 主进程循环：broadcast `("step",)` 给所有 worker；每 worker 各自执行自己的 `LLMEngine.step()`，返回本步完成的 seq `[(gid, completion_token_ids)]` 和 `num_tokens`
3. 主进程按 gid 聚合 outputs；更新 tqdm；broadcast `is_finished`，全部 True 才退出循环
4. 结束后 broadcast `metrics_snapshot`，合并 N 份为聚合 `EngineMetrics`；decode outputs 返回给用户

### 3.3 与现有架构的正交性

- `LLMEngine` / `Scheduler` / `BlockManager` / `ModelRunner` / `Sequence` / `MetricsCollector` 本体**完全不改**
- `DPLLMEngine` 和 `LLMEngine` 是**平级**的门面类，对外暴露相同 API
- `LLM` 门面（在 `nanovllm/llm.py`）在 `__new__` 里根据 `data_parallel_size` 分派到 `LLMEngine` 或 `DPLLMEngine`

## 4. 核心改动

### 4.1 文件清单

**新建：**

- `nanovllm/engine/dp_engine.py`（预估 200~280 行）
  - `DPLLMEngine` 类：主进程实体
  - `_dp_worker_entry(rank, config, pipe)`：worker 进程 entry function

**修改：**

- `nanovllm/llm.py`：`LLM` 从"直接继承 LLMEngine"改为"`__new__` 工厂"
- `nanovllm/config.py`：新增 `data_parallel_size: int = 1` 字段 + `__post_init__` 校验
- `nanovllm/engine/model_runner.py`：1 行改动——`init_process_group` 的 URL 从硬编码 `"tcp://localhost:2333"` 改为读 `NANO_VLLM_DIST_PORT` 环境变量（见 §4.4）

**明确不改：**

- `nanovllm/engine/llm_engine.py`（worker 内作为组件直接复用）
- `nanovllm/engine/scheduler.py` / `block_manager.py` / `sequence.py`
- `nanovllm/engine/metrics.py`（`MetricsCollector` / `EngineMetrics` 不变）
- `nanovllm/layers/**` / `nanovllm/models/**` / `nanovllm/utils/**`
- `tests/conftest.py`（session `llm` fixture 保留）
- `bench.py`（可选修改传 `data_parallel_size`，不是本 spec scope）

### 4.2 Config 校验

```python
# nanovllm/config.py __post_init__ 新增：
assert self.data_parallel_size >= 1
assert not (self.tensor_parallel_size > 1 and self.data_parallel_size > 1), \
    "DP × TP 暂不支持，是下一项独立 spec"
if self.data_parallel_size > 1:
    assert self.data_parallel_size <= torch.cuda.device_count(), \
        f"data_parallel_size={self.data_parallel_size} > 可用 GPU 数"
```

### 4.3 LLM 门面

```python
# nanovllm/llm.py
from nanovllm.engine.llm_engine import LLMEngine
from nanovllm.engine.dp_engine import DPLLMEngine

class LLM:
    def __new__(cls, model, **kwargs):
        if kwargs.get("data_parallel_size", 1) > 1:
            return DPLLMEngine(model, **kwargs)
        return LLMEngine(model, **kwargs)
```

**注意**：去掉原本 `class LLM(LLMEngine): pass` 的继承关系。`LLM` 不再是类型，而是 dispatch 工厂；对外类型是 `LLMEngine` 或 `DPLLMEngine`。测试或用户代码里 `isinstance(llm, LLMEngine)` 检查需要相应调整；本项目当前没有这种用法（已 grep 确认）。

### 4.4 model_runner.py 的端口 env 化（1 行改动）

`model_runner.py:26` 当前硬编码：

```python
dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)
```

改为读环境变量（`import os` 加到文件头）：

```python
_dist_port = os.environ.get("NANO_VLLM_DIST_PORT", "2333")
dist.init_process_group("nccl", f"tcp://localhost:{_dist_port}", world_size=self.world_size, rank=rank)
```

**为什么必须**：DP worker 是 N 个独立 Python 进程，若都 bind 同一 `tcp://localhost:2333` 做 NCCL rendezvous 就端口冲突。通过 env var 让每个 worker（在 entry 最开始）设不同端口（`2333 + rank`）规避。**不影响**单机 DP=1 / TP>1 行为：env 未设时 fallback 原默认端口。

## 5. 通信协议

### 5.1 IPC 机制

- `torch.multiprocessing.Pipe(duplex=True)`：主进程持 `parent_conn`，worker 持 `child_conn`
- **同步 RPC 语义**：主 `send(msg)` 后必 `recv()`；worker 严格"收一发一"，不做流水线
- 序列化：Python 默认 pickle；`SamplingParams` 是 dataclass，`Sequence` 本身不跨进程传（只传 token_ids + sp）

### 5.2 消息集

| 方法 | 参数 | 返回值 |
|---|---|---|
| `add` | `gid, token_ids, sampling_params` | `None`（ack） |
| `step` | — | `(finished_outputs, num_tokens, is_prefill)`；`finished_outputs = [(gid, completion_token_ids), ...]` |
| `is_finished` | — | `bool` |
| `metrics_snapshot` | — | `dict`：`{step_samples, completed_requests, total_preemptions}` |
| `reset_metrics` | — | `None` |
| `exit` | — | `None`；worker 收到后 break loop、进程正常退出 |

### 5.3 Worker loop 框架

```python
def _dp_worker_entry(rank, config, child_conn):
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)  # 必须在 import torch 前
    os.environ["NANO_VLLM_DIST_PORT"] = str(2333 + rank)  # 避让 init_process_group 端口冲突
    # 注：由于 spawn context，父进程的 torch import 不被继承
    from nanovllm.engine.llm_engine import LLMEngine
    engine = LLMEngine(config.model, **config_kwargs_tp1)
    while True:
        method, *args = child_conn.recv()
        if method == "exit":
            child_conn.send(None)
            break
        result = _dispatch(engine, method, args)
        child_conn.send(result)
    engine.exit()
```

## 6. 全局 Seq ID

### 6.1 问题

`nanovllm/engine/sequence.py` 里 `Sequence.counter` 是 class-level `itertools.count(0)`，每个进程独立自增。N 个 worker 各自从 0 开始 → 不同 worker 生成相同 seq_id，主进程 outputs dict 会冲突。

### 6.2 方案

主进程维护 `self.next_global_id: int = 0`，`add_request` 时分配 `gid = next_global_id; next_global_id += 1`，通过 pipe 传给 worker。Worker 构造 Sequence 后 override：

```python
def _handle_add(engine, gid, token_ids, sp):
    seq = Sequence(token_ids, sp)
    seq.seq_id = gid                # override 掉 class counter 生成的 id
    seq.arrival_time = perf_counter()
    engine.scheduler.add(seq)
```

Worker step 完成后的 outputs 用 `seq.seq_id`（= gid），主进程按 gid 聚合；不同 worker 的 gid 不重叠因为是全局分配。

**注意**：`MetricsCollector.completed_requests` dict 的 key 是 seq_id，用 gid 保证跨 worker 合并时不冲突。

## 7. Metrics 聚合

### 7.1 单边实现

Worker 内 `LLMEngine.metrics` 正常累积 step_samples / completed_requests / total_preemptions。`metrics_snapshot` 返回三份原生可 pickle 的数据：

```python
def _handle_metrics_snapshot(engine):
    m = engine.metrics
    return {
        "step_samples": list(m.step_samples),          # list[StepSample]
        "completed_requests": dict(m.completed_requests),  # dict[int, RequestMetrics]
        "total_preemptions": m.total_preemptions,
    }
```

### 7.2 主进程聚合

```python
def get_aggregate_metrics(self) -> EngineMetrics:
    from nanovllm.engine.metrics import MetricsCollector
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

### 7.3 时钟与语义

- `perf_counter` 在同一物理机上对所有进程一致（monotonic clock），合并 step_samples 时 wall time 在同一刻度
- 不做时钟同步修正；毫秒级漂移对 summary percentile 无可见影响
- `reset_metrics()` 广播 `("reset_metrics",)` 给所有 worker，语义等同于单 engine 场景的整体清零

## 8. 边界情况

| 情况 | 处理 |
|---|---|
| `data_parallel_size == 1` | `LLM.__new__` 走 `LLMEngine`，不启动 worker 进程，零退化 |
| `data_parallel_size > torch.cuda.device_count()` | `Config.__post_init__` assert 失败 |
| `tensor_parallel_size>1 and data_parallel_size>1` | `Config.__post_init__` assert 失败 |
| Worker 进程崩溃 | 主进程 `pipe.recv()` 抛 `EOFError`，整体失败退出（不做 restart） |
| CUDA 上下文隔离 | `CUDA_VISIBLE_DEVICES` 必须在 `import torch` 前设；用 `spawn` context 保证子进程从干净 Python 起 |
| `init_process_group` 端口冲突 | 每个 worker 通过 `NANO_VLLM_DIST_PORT` env var 用不同端口（`2333 + rank`）；需改 `model_runner.py:26` 让 URL 读 env（§4.4，1 行改动） |
| `exit()` 幂等 | 首次调用发 exit 消息 + join，后续空转；参考 `LLMEngine.exit()` 的 flag 模式 |
| `atexit` 注册 | `DPLLMEngine` 也注册 `atexit.register(self.exit)`，和 `LLMEngine` 行为一致 |
| Tokenizer | 只主进程 load；worker 的 LLMEngine 也会 load 一份（当前 LLMEngine `__init__` 里有 `AutoTokenizer.from_pretrained`）——有冗余但简单，不在本 spec 优化 |
| Round-robin 负载倾斜 | 接受；256 条 seq 量级的 bench 场景下偏差 <10%（中心极限），对吞吐评估足够 |
| Prefix cache 不跨 worker | 接受；YAGNI。同 prompt 打到不同 worker 各自重建 |

## 9. 测试策略

### 9.1 单元测试（无 GPU）

- `tests/test_dp_dispatch.py`（预估 ~100 行）
  - Round-robin counter：3 条 add 分别打到 rank 0/1/2；第 4 条回 rank 0
  - Global seq_id 严格递增、无重复
  - Mock pipe（用 `mp.Pipe` + stub 函数当 worker）验证消息收发正确

- `tests/test_dp_metrics_aggregation.py`（预估 ~100 行）
  - 手工构造 2~3 份 `metrics_snapshot` dict
  - 验证合并后：step_samples 长度 = 求和、request_metrics 合并不丢不重、preemptions_total 求和正确
  - 不涉及 worker / GPU

### 9.2 E2E 活性（需 ≥2 GPU）

- `tests/test_dp_e2e.py`（预估 ~80 行）
  - Fixture: `DPLLMEngine(model, data_parallel_size=2, enforce_eager=True, tensor_parallel_size=1)`（avoid CUDA graph + TP 组合复杂度）
  - 用例 1：`generate` 16 条混合长度 prompt；断言所有 output 非空、`metrics.ttft > 0`
  - 用例 2：`get_aggregate_metrics().step_samples` 总步数 > 0；至少能看出 prefill step + decode step
  - 用例 3：`reset_metrics()` 后再跑一次，`step_samples` 不累积
  - Teardown：显式 `exit()` 释放所有 worker

### 9.3 Bench sanity（非自动化）

- 跑 `python bench.py`（1 卡 DP=1）vs `DPLLMEngine(..., data_parallel_size=2)` 版本
- 期望 2 卡吞吐 ≈ 1.8~2.0× 单卡（受 IPC/dispatch overhead 影响，不可能完全 2×）
- 手工眼看，不断言

### 9.4 不做

- 不做 Qwen3-0.6B 以外模型的兼容性测试
- 不做 `data_parallel_size=3` 强制测试（3×4090 场景可选，硬件支持才跑）

## 10. 开销与风险

### 10.1 开销

- **IPC**：每 step 2×N 次 pipe round-trip；pickle 序列化 outputs list 每条 ~几十 B。总开销 <1 ms / step，相对单步 GPU 计算 10~50 ms 可忽略
- **显存**：每 worker 独立 model weights + KV cache；3 × 单卡显存总量（不共享）
- **启动时间**：N 个 worker 并行 load weights + warmup + CUDA graph 估算 KV blocks；墙钟略慢于单 engine 但可接受
- **Tokenizer 冗余**：每 worker 的 `LLMEngine.__init__` 会独立 load tokenizer 一份，N=3 时内存多占 ~100 MB；不在本 spec 优化范围

### 10.2 风险

| 风险 | 缓解 |
|---|---|
| `init_process_group` 端口冲突 | worker 启动前设 `NANO_VLLM_DIST_PORT = str(2333 + rank)`；`model_runner.py:26` URL 从 env 读（§4.4） |
| `CUDA_VISIBLE_DEVICES` 设置时机 | 严格用 `mp.get_context("spawn")`；env 设置在 worker entry 第一行、`import torch` 之前 |
| Metrics `reset_metrics()` 与进行中 step 的竞态 | 严格 sync 节拍下 `reset_metrics()` 只在 `is_finished()` 返回 True 后调用；调用语义和单 engine 一致 |
| Worker 崩溃不可恢复 | 接受；本 spec YAGNI 不做 restart |
| `atexit` 触发顺序 | 主进程 `exit()` 幂等化；`atexit.register` 确保异常退出也能清理子进程 |

## 11. 对后续项目的铺垫

本 spec 完成后，roadmap 第 5 项（候选）是 **KV offload**。DP 的多进程架构对 KV offload 透明——offload 是 per-engine 的内存系统优化，与 DP 的请求分发正交。将来若做 DP × TP 组合，当前的 "每 worker 一个 LLMEngine(TP=1)" 架构需扩展为 "每 DP rank 一个子 process group"，但请求分发 / metrics 聚合层可以完整复用。
