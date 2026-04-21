# Chunked Prefill Generalization · 设计文档

- **日期**：2026-04-21
- **作者**：Orange-Bei（与 Claude brainstorming）
- **状态**：Draft，待评审

---

## 1. 背景

`nanovllm/engine/scheduler.py::schedule` 的 prefill 循环当前包含一条显式限制：

```python
if remaining < num_tokens and scheduled_seqs:    # no budget
    # only allow chunked prefill for the first seq
    break
```

效果：一步调度里只有**排在 waiting 队首的第一条 seq** 允许被 chunk。一旦已经有至少一条 seq 被纳入本步 `scheduled_seqs`，后续任何 seq 如果超预算就直接让它等下一步，不 chunk。

限制带来的浪费：当 `waiting` 队列混合了长 prefill 与短 prefill，长的先排完后，budget 可能只剩很小余量；下一条 seq（长或短）只要不能完整装下就被跳过，这部分 budget 就 idle。实测 `StepSample.num_batched_tokens` 在混合负载下存在明显的"留白"。

本改动是二次开发 roadmap 第 3 项，目标：**去掉 "first-seq only" 限制，让任意 seq 都能被 chunk**，把 budget 填满。

## 2. 目标与非目标

### 2.1 目标

- **任意 seq 可 chunk**：不论是 waiting 队首还是后续 seq，只要本步剩余 budget 不足以容纳其全部 prefill 需求，就按 `remaining` 做一步 chunk，剩余部分留到下一步继续
- **零底层改动**：不动 `ModelRunner` / `BlockManager` / `Sequence` / `context.py` / attention / CUDA graph / 任何 layer 代码
- **不变量保持**：`num_batched_tokens <= max_num_batched_tokens`、`len(scheduled_seqs) <= max_num_seqs`、prefill 优先级高于 decode 的 dispatch 策略都不变
- **向后兼容**：所有现有行为（非 chunk 的快路径、单 seq chunk 的回归路径、prefix cache、抢占与重跑）语义不变

### 2.2 非目标（YAGNI）

- 不做 prefill + decode 同批的连续批处理（continuous batching 式混合）
- 不加 per-seq 优先级、公平性配额、max_chunks_per_step 等策略——当前的"先到先服务、一步填满 budget"就够
- 不新增可观测字段——现有 `StepSample.num_batched_tokens` 可直接观测填充率提升
- 不做自动化性能回归断言——改动足够小，手工 `bench.py` sanity check 即可

## 3. 架构决策

**决策：删除一行 break 即完成功能。**

| 决策点 | 选择 | 理由 |
|---|---|---|
| 推广范围 | 仅 scheduler，不碰底层 | `prepare_prefill` 的 `cu_seqlens_q` / `num_cached_tokens` / `start = min(num_cached_tokens, seqlen-1)` 已支持任意 chunk 位置；flash-attn varlen + prefix cache 路径已就位 |
| 是否需要新策略 | 否 | "每步最多一条 seq 处于 in-progress chunk"的结构天然由 `remaining=0 → 下轮 break` 保证，无需额外控制 |
| 是否混合 prefill/decode | 否 | 架构变动太大，当前 prefill-first / decode-next 的 two-phase dispatch 保留 |
| 如何验证 | 单测 + e2e 活性 | 单测覆盖调度逻辑的全部场景；e2e 只验证不崩、输出合理 |

## 4. 核心改动

### 4.1 `scheduler.py::schedule` 的 prefill 循环

改动前（关键段）：

```python
while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
    seq = self.waiting[0]
    num_tokens = max(seq.num_tokens - seq.num_cached_tokens, 1)
    remaining = self.max_num_batched_tokens - num_batched_tokens
    if remaining == 0 or (not seq.block_table and not self.block_manager.can_allocate(seq)):
        break
    if remaining < num_tokens and scheduled_seqs:    # no budget
        # only allow chunked prefill for the first seq
        break
    if not seq.block_table:
        self.block_manager.allocate(seq)
    if seq.first_scheduled_time is None:
        seq.first_scheduled_time = perf_counter()
    seq.num_scheduled_tokens = min(num_tokens, remaining)
    if seq.num_scheduled_tokens == num_tokens:
        seq.status = SequenceStatus.RUNNING
        self.waiting.popleft()
        self.running.append(seq)
    scheduled_seqs.append(seq)
    num_batched_tokens += seq.num_scheduled_tokens
```

改动后：**删除** `if remaining < num_tokens and scheduled_seqs: break` 那 3 行。

```python
while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
    seq = self.waiting[0]
    num_tokens = max(seq.num_tokens - seq.num_cached_tokens, 1)
    remaining = self.max_num_batched_tokens - num_batched_tokens
    if remaining == 0 or (not seq.block_table and not self.block_manager.can_allocate(seq)):
        break
    if not seq.block_table:
        self.block_manager.allocate(seq)
    if seq.first_scheduled_time is None:
        seq.first_scheduled_time = perf_counter()
    seq.num_scheduled_tokens = min(num_tokens, remaining)
    if seq.num_scheduled_tokens == num_tokens:
        seq.status = SequenceStatus.RUNNING
        self.waiting.popleft()
        self.running.append(seq)
    scheduled_seqs.append(seq)
    num_batched_tokens += seq.num_scheduled_tokens
```

**为什么只改这一处就足够：**

- `min(num_tokens, remaining)` 已经是 chunk 的实现（当 `remaining < num_tokens` 时自动截短）
- 本步 chunk 后，`num_batched_tokens += remaining`，下一轮 `remaining = 0`，命中外层 `break`——没有死循环
- 被 chunk 的 seq `num_scheduled_tokens < num_tokens`，`status` 仍为 `WAITING`，`waiting.popleft()` 不触发——下步调度时仍从队首拿到它继续 chunk
- `first_scheduled_time` 在首次进入 scheduled 时设置，不再重置——TTFT 语义保持"首次被调度 → 首个 completion token"

### 4.2 不变量（改动前后恒成立）

| 不变量 | 说明 |
|---|---|
| `num_batched_tokens <= max_num_batched_tokens` | 总 token 预算上限 |
| `len(scheduled_seqs) <= max_num_seqs` | 并发 seq 上限 |
| 每步至多一条 seq 处于"in-progress chunk" | 由"一旦 chunk 则 remaining=0 → 下轮 break"结构保证 |
| prefill 调度优先级 > decode 调度 | `if scheduled_seqs: return True` 段不动，prefill 有选时决不掉到 decode |
| 被 chunk 的 seq 留在 `waiting[0]`，`status=WAITING`，下步从同一位置继续 | `popleft` 只发生在完整 prefill 完成时 |

## 5. 边界情况

| 情况 | 处理 |
|---|---|
| 首条 seq 连 block 都分配不出 | 原 `can_allocate` break 仍生效，行为不变 |
| `num_cached_tokens == num_tokens`（前缀 100% 命中） | `num_tokens = max(0, 1) = 1`，chunk 至多 1 token 前向，正常走 |
| 抢占（`preempt`） | `preempt` 只对 `running` 队列（正在 decode）的 seq 触发；chunked prefill 的 seq 始终在 `waiting` 队列，不会被抢占。被抢占的 decode seq 回到 waiting 状态、重新 prefill 时（`num_completion_tokens > 0` 分支），也同样受益于新行为，可以被 chunk |
| Prefix cache 与 chunk 交互 | 前缀命中只在 `allocate` 时发生（首次进 scheduled）；chunk 中间步骤 `seq.block_table` 已存在，不再走 `allocate` 路径 |
| `max_num_batched_tokens == 0` | Config 构造时就不合法（warmup 会炸），不在本 spec 处理范围 |
| CUDA graph decode 批次 | 与 prefill 调度互斥（`if scheduled_seqs from prefill: return True`），不受影响 |
| TP `world_size > 1` | Sequence 状态传递走 `__getstate__`（不含调度相关字段），worker rank 只接 `(input_ids, positions, is_prefill)`，调度逻辑全在 rank 0，无影响 |

## 6. 测试策略

### 6.1 单元测试 `tests/test_scheduler_chunked_prefill.py`（无 GPU）

构造真实 `Scheduler` + 真实 `BlockManager`（纯 Python）+ 真实 `MetricsCollector`（纯 Python）。用真实 `Sequence`（可以传 mock `SamplingParams`）。

**覆盖场景：**

| 场景 | 设置 | 断言 |
|---|---|---|
| A. 回归：全短 seq 一步完成 | 3 条各 100 token，`max_num_batched_tokens=8192` | `len(scheduled_seqs)==3`、无 chunk（所有 `num_scheduled_tokens == num_tokens`） |
| B. 回归：单长 seq 被 chunk | 1 条 10000 token，budget 8192 | `len(scheduled_seqs)==1`、`scheduled_seqs[0].num_scheduled_tokens == 8192`、`status==WAITING` |
| C. 新行为：多 seq 混合被 chunk | seq0=3000、seq1=5000、seq2=2000，budget=8192 | seq0+seq1 完整入列、seq2 被 chunk 到 192；`num_batched_tokens==8192`；seq2 `status==WAITING`、仍在 `waiting[0]` |
| D. 连续步进 chunk | 接 C 的场景，模拟一步完成后调 `postprocess` 和下一次 `schedule` | seq2 继续被 chunk 或完成；总 token 守恒 |
| E. 不变量属性 | 任意合法输入 | `num_batched_tokens <= max_num_batched_tokens`、`len(scheduled_seqs) <= max_num_seqs` 恒成立 |

### 6.2 E2E 活性 `tests/test_chunked_prefill_e2e.py`（GPU + 模型）

**独立 fixture**（不用 `conftest.py` 的 session 共享 LLM，因为需要不同 `max_num_batched_tokens`）：

```python
@pytest.fixture(scope="module")
def chunked_llm():
    from nanovllm import LLM
    instance = LLM(
        MODEL_PATH,
        enforce_eager=True,
        max_model_len=1024,
        max_num_batched_tokens=256,   # 小预算强制 chunk
        tensor_parallel_size=1,
    )
    yield instance
    instance.exit()   # 显式释放 dist，供后续 module 创建 session LLM
```

**测试用例：**

- 构造 prompt 使其 tokenize 后长度 > 256（例如 `"hello " * 200`）
- 调 `llm.generate(...)`，断言：
  - 不抛异常
  - 返回 `output["text"]` 非空
  - `output["metrics"].ttft > 0`
  - `output["metrics"].num_prompt_tokens > 256`（确认 prompt 确实超预算）
  - `llm.get_aggregate_metrics().step_samples` 中存在 `num_batched_tokens == 256` 的 step（确认 budget 被打满、chunk 生效）

### 6.3 不做

- 不做自动化性能回归断言——改动足够小（一行），bench.py 手工跑一次 sanity 即可
- 不在单测里 mock GPU/attention——本改动只碰 scheduler，不需要

## 7. 文件清单

**修改**

- `nanovllm/engine/scheduler.py`：删除 prefill 循环里的 3 行（`if remaining < num_tokens and scheduled_seqs:` 那个 break 块）

**新建**

- `tests/test_scheduler_chunked_prefill.py`（~150 行）
- `tests/test_chunked_prefill_e2e.py`（~60 行，含独立 `chunked_llm` fixture）

**明确不改**

- `nanovllm/engine/model_runner.py` / `block_manager.py` / `sequence.py` / `metrics.py` / `llm_engine.py` / `llm.py`
- `nanovllm/layers/**` / `nanovllm/models/**` / `nanovllm/utils/**` / `nanovllm/sampling_params.py`
- `tests/conftest.py`（session `llm` fixture 保留不动）
- `example.py` / `bench.py` / `pyproject.toml`

## 8. 开销分析

- **改动成本**：scheduler 一行删除，零运行时开销变化
- **调度器每步开销**：原来命中 "first-seq only" break 时提前退出循环；新行为多走 1-2 次循环迭代（把最后一条 seq chunk 进去），每次迭代是几个 Python attribute 读 + 一次 `min`——纳秒级，可忽略
- **吞吐预期提升**：取决于 waiting 队列的长度分布。混合负载（长短 prefill 交错）下，`StepSample.num_batched_tokens` 的 p50 应该上升，整体吞吐 +5~15%（定性估算，实测以 bench 为准）
- **延迟影响**：个别被 chunk 的 seq 的 TTFT 略增（多等一步），但整体队列尾延迟下降（等待的 seq 更快被纳入）

## 9. 对后续项目的铺垫

- **DP 并行（#4）**：调度层不新增状态，DP 协调层仍可把单副本 Scheduler 作为基本单元、在副本间做 round-robin。本改动对 DP 层透明
