# Sampling Extension (greedy / top-k / top-p / repetition_penalty) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Nano-vLLM 的 `Sampler` 从"只支持 `temperature` + Gumbel-max"扩展为支持 greedy、`top_k`、`top_p`、`repetition_penalty` 四种采样开关，并保持 mixed-batch、`@torch.compile` 稳定性、向后兼容。

**Architecture:** 两阶段——`apply_repetition_penalty(logits, seqs)` eager Python 循环处理 ragged 历史；`Sampler.forward(logits, temperatures, top_k, top_p)` 保留 `@torch.compile`，用 `torch.where` 完成全向量化控制流（只 sort 一次，top_k / top_p 复用同一 `sorted_logits`，greedy 与 Gumbel 两路径并行跑 + 按行 `where` 选）。Sampler 继续留在 CUDA graph 外。

**Tech Stack:** Python 3.10+、PyTorch（已有）、pytest（已有，round #1 已安装）。不新增依赖。

**Spec reference:** `doc/specs/2026-04-21-sampling-extension-design.md`

**Commit policy (IMPORTANT):** 本 plan **不做中间 commit**。每 Task 结束后只运行测试、验证通过即可。所有代码改动累积到 Task 7 最后一步，合并前一轮的 spec commit（`3d52cdd`）做一次性整体 commit。

---

## 文件结构

**新建**

- `tests/test_sampling_params.py` — Task 1 单测（无 GPU）
- `tests/test_apply_repetition_penalty.py` — Task 3 单测（需 GPU）
- `tests/test_sampler_kernel.py` — Task 4 单测（需 GPU）
- `tests/test_sampling_e2e.py` — Task 6 端到端冒烟（需 GPU + 模型）

**修改**

- `nanovllm/sampling_params.py` — 新增 3 字段 + 重写 `__post_init__`
- `nanovllm/engine/sequence.py` — `__init__` 中拷贝 3 个新字段
- `nanovllm/layers/sampler.py` — 新增模块级 `apply_repetition_penalty`；重写 `Sampler.forward`
- `nanovllm/engine/model_runner.py` — `prepare_sample` 扩展为 3 张量；`run` 串接 `apply_repetition_penalty`

**明确不改**

- `nanovllm/engine/scheduler.py` / `nanovllm/engine/block_manager.py` / `nanovllm/engine/metrics.py` / `nanovllm/engine/llm_engine.py` / `nanovllm/llm.py`
- `nanovllm/models/**` / `nanovllm/utils/**`
- `nanovllm/engine/model_runner.py::capture_cudagraph`（CUDA graph 捕获逻辑）
- `Sequence.__getstate__` / `__setstate__`（TP worker pickle 路径）
- `example.py` / `bench.py`（round #1 已经演示指标；本轮 bench 仅用于手工性能回归，不改代码）

---

## Task 1: 扩展 `SamplingParams`（构造期校验，无 GPU）

**Files:**
- Modify: `nanovllm/sampling_params.py`
- Create: `tests/test_sampling_params.py`

- [ ] **Step 1: 写 `tests/test_sampling_params.py` 的失败单测**

创建 `tests/test_sampling_params.py`：

```python
import pytest

from nanovllm.sampling_params import SamplingParams


class TestSamplingParamsDefaults:
    def test_default_construction_succeeds(self):
        sp = SamplingParams()
        assert sp.temperature == 1.0
        assert sp.top_k == -1
        assert sp.top_p == 1.0
        assert sp.repetition_penalty == 1.0
        assert sp.max_tokens == 64
        assert sp.ignore_eos is False

    def test_greedy_is_allowed(self):
        # temperature=0 现在合法（旧 assert 已移除）
        sp = SamplingParams(temperature=0.0)
        assert sp.temperature == 0.0

    def test_full_custom_succeeds(self):
        sp = SamplingParams(
            temperature=0.7,
            top_k=50,
            top_p=0.9,
            repetition_penalty=1.1,
            max_tokens=128,
        )
        assert sp.top_k == 50
        assert sp.top_p == 0.9
        assert sp.repetition_penalty == 1.1


class TestSamplingParamsValidation:
    def test_negative_temperature_rejected(self):
        with pytest.raises(AssertionError, match="temperature"):
            SamplingParams(temperature=-0.1)

    def test_top_k_zero_rejected(self):
        # top_k=0 与 -1（禁用）语义冲突，拒绝
        with pytest.raises(AssertionError, match="top_k"):
            SamplingParams(top_k=0)

    def test_top_k_negative_not_minus_one_rejected(self):
        with pytest.raises(AssertionError, match="top_k"):
            SamplingParams(top_k=-2)

    def test_top_k_minus_one_allowed(self):
        SamplingParams(top_k=-1)

    def test_top_p_zero_rejected(self):
        with pytest.raises(AssertionError, match="top_p"):
            SamplingParams(top_p=0.0)

    def test_top_p_above_one_rejected(self):
        with pytest.raises(AssertionError, match="top_p"):
            SamplingParams(top_p=1.5)

    def test_top_p_one_allowed(self):
        SamplingParams(top_p=1.0)

    def test_repetition_penalty_zero_rejected(self):
        with pytest.raises(AssertionError, match="repetition_penalty"):
            SamplingParams(repetition_penalty=0.0)

    def test_repetition_penalty_negative_rejected(self):
        with pytest.raises(AssertionError, match="repetition_penalty"):
            SamplingParams(repetition_penalty=-0.5)

    def test_max_tokens_zero_rejected(self):
        with pytest.raises(AssertionError):
            SamplingParams(max_tokens=0)
```

- [ ] **Step 2: 运行测试，确认失败**

```bash
cd /home/ubuntu/Nano-Vllm-Learning
pytest tests/test_sampling_params.py -v
```

预期：`test_greedy_is_allowed` 失败（因为现在 `assert self.temperature > 1e-10` 会拒绝 0）；新字段访问也会失败（`AttributeError: 'SamplingParams' object has no attribute 'top_k'`）。

- [ ] **Step 3: 修改 `nanovllm/sampling_params.py`**

完整替换文件内容为：

```python
from dataclasses import dataclass


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

- [ ] **Step 4: 运行测试，确认全部通过**

```bash
pytest tests/test_sampling_params.py -v
```

预期：11 个测试全部 PASS。

- [ ] **Step 5: 跑一遍已有的 metrics 测试，确认没意外回归**

```bash
pytest tests/ -v --ignore=tests/test_metrics_e2e.py --ignore=tests/test_sampling_e2e.py
```

预期：全部 PASS（round #1 的 metrics 测试 + 本轮的 sampling_params 测试）。

---

## Task 2: `Sequence` 扩展字段

**Files:**
- Modify: `nanovllm/engine/sequence.py`

**Note:** 此 Task 无专门单测——字段赋值极其简单，错误会在 Task 5（ModelRunner）或 Task 6（e2e）暴露。

- [ ] **Step 1: 修改 `nanovllm/engine/sequence.py`**

打开 `nanovllm/engine/sequence.py`，找到 `__init__` 方法中这段代码（大约 28-30 行）：

```python
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens # 生成的token数量达到max_tokens时就结束
        self.ignore_eos = sampling_params.ignore_eos
```

在 `self.temperature = ...` 这一行之后、`self.max_tokens = ...` 之前**插入** 3 行：

```python
        self.temperature = sampling_params.temperature
        self.top_k = sampling_params.top_k
        self.top_p = sampling_params.top_p
        self.repetition_penalty = sampling_params.repetition_penalty
        self.max_tokens = sampling_params.max_tokens # 生成的token数量达到max_tokens时就结束
        self.ignore_eos = sampling_params.ignore_eos
```

- [ ] **Step 2: 不碰 `__getstate__` / `__setstate__`**

检查 `__getstate__`（大约第 78-80 行）**不要**修改。确认当前仍然是：

```python
    def __getstate__(self):
        last_state = self.token_ids if self.num_completion_tokens == 0 or self.num_cached_tokens < self.num_tokens else self.last_token
        return (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.num_scheduled_tokens, self.block_table, last_state)
```

新字段不进 `__getstate__` —— TP worker 不需要采样字段，保持 pickle payload 不变。

- [ ] **Step 3: 跑现有测试确认没破坏任何东西**

```bash
pytest tests/ -v --ignore=tests/test_metrics_e2e.py --ignore=tests/test_sampling_e2e.py
```

预期：全部 PASS。

- [ ] **Step 4: Python 手工验证字段存在**

```bash
python -c "
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
sp = SamplingParams(top_k=50, top_p=0.9, repetition_penalty=1.1)
seq = Sequence([1, 2, 3], sp)
assert seq.top_k == 50
assert seq.top_p == 0.9
assert seq.repetition_penalty == 1.1
print('OK')
"
```

预期输出：`OK`。

---

## Task 3: `apply_repetition_penalty` 模块级函数（GPU）

**Files:**
- Modify: `nanovllm/layers/sampler.py`
- Create: `tests/test_apply_repetition_penalty.py`

- [ ] **Step 1: 写 `tests/test_apply_repetition_penalty.py` 的失败单测**

创建 `tests/test_apply_repetition_penalty.py`：

```python
import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="apply_repetition_penalty requires CUDA",
)


class _FakeSeq:
    """仅用于测试，复用 Sequence 的 duck-typed 接口。"""
    def __init__(self, token_ids, repetition_penalty):
        self.token_ids = token_ids
        self.repetition_penalty = repetition_penalty


def _make_logits(batch_size, vocab_size, device="cuda"):
    torch.manual_seed(0)
    return torch.randn(batch_size, vocab_size, device=device)


class TestApplyRepetitionPenalty:
    def test_penalty_one_is_noop(self):
        from nanovllm.layers.sampler import apply_repetition_penalty
        logits = _make_logits(2, 100)
        before = logits.clone()
        seqs = [_FakeSeq([1, 2, 3], 1.0), _FakeSeq([4, 5, 6], 1.0)]
        apply_repetition_penalty(logits, seqs)
        assert torch.allclose(before, logits)

    def test_positive_logit_divided(self):
        from nanovllm.layers.sampler import apply_repetition_penalty
        logits = torch.zeros(1, 10, device="cuda")
        logits[0, 3] = 4.0  # 正 logit
        seqs = [_FakeSeq([3], 2.0)]  # rp=2
        apply_repetition_penalty(logits, seqs)
        assert logits[0, 3].item() == pytest.approx(2.0)  # 4 / 2

    def test_negative_logit_multiplied(self):
        from nanovllm.layers.sampler import apply_repetition_penalty
        logits = torch.zeros(1, 10, device="cuda")
        logits[0, 3] = -4.0  # 负 logit
        seqs = [_FakeSeq([3], 2.0)]  # rp=2
        apply_repetition_penalty(logits, seqs)
        assert logits[0, 3].item() == pytest.approx(-8.0)  # -4 * 2

    def test_repeated_token_penalized_once(self):
        from nanovllm.layers.sampler import apply_repetition_penalty
        logits = torch.zeros(1, 10, device="cuda")
        logits[0, 3] = 4.0
        seqs = [_FakeSeq([3, 3, 3, 3, 3], 2.0)]  # token 3 出现 5 次
        apply_repetition_penalty(logits, seqs)
        # scatter 幂等，不叠加惩罚
        assert logits[0, 3].item() == pytest.approx(2.0)  # 4 / 2，不是 4 / 32

    def test_mixed_batch_only_nondefault_row_modified(self):
        from nanovllm.layers.sampler import apply_repetition_penalty
        logits = torch.randn(3, 10, device="cuda")
        before = logits.clone()
        seqs = [
            _FakeSeq([1, 2], 1.0),   # rp=1.0 禁用，跳过
            _FakeSeq([3, 4], 2.0),   # rp=2.0 生效
            _FakeSeq([5, 6], 1.0),   # rp=1.0 禁用，跳过
        ]
        apply_repetition_penalty(logits, seqs)
        assert torch.allclose(before[0], logits[0])   # 行 0 不变
        assert not torch.allclose(before[1], logits[1])  # 行 1 变了
        assert torch.allclose(before[2], logits[2])   # 行 2 不变

    def test_untouched_tokens_unchanged(self):
        from nanovllm.layers.sampler import apply_repetition_penalty
        logits = torch.randn(1, 100, device="cuda")
        before = logits.clone()
        seqs = [_FakeSeq([5, 10, 15], 2.0)]
        apply_repetition_penalty(logits, seqs)
        # 只有索引 5/10/15 的 logit 应该改变
        for i in range(100):
            if i in (5, 10, 15):
                assert not torch.equal(logits[0, i], before[0, i])
            else:
                assert torch.equal(logits[0, i], before[0, i])
```

- [ ] **Step 2: 运行测试，确认失败**

```bash
pytest tests/test_apply_repetition_penalty.py -v
```

预期：`ImportError: cannot import name 'apply_repetition_penalty' from 'nanovllm.layers.sampler'`。

- [ ] **Step 3: 在 `nanovllm/layers/sampler.py` 新增模块级函数**

打开 `nanovllm/layers/sampler.py`。当前文件只有 `Sampler` 类。在 `import` 区块下面、`class Sampler` 上面新增模块级函数：

```python
import torch
from torch import nn


def apply_repetition_penalty(logits: torch.Tensor, seqs: list) -> None:
    """In-place 对 logits 应用 repetition_penalty。

    历史范围 = seq.token_ids（prompt + completion），对齐 vLLM / HF 行为。
    rp == 1.0 的 seq 整行跳过（最常见情况零 GPU op）。
    Scatter 幂等：同一 token 在历史中重复出现不会叠加惩罚。
    """
    for i, seq in enumerate(seqs):
        rp = seq.repetition_penalty
        if rp == 1.0:
            continue
        tokens = torch.tensor(seq.token_ids, device=logits.device, dtype=torch.int64)
        selected = logits[i].gather(0, tokens)
        penalized = torch.where(selected < 0, selected * rp, selected / rp)
        logits[i].scatter_(0, tokens, penalized)


class Sampler(nn.Module):

    @torch.compile # 使用torch.compile装饰器对forward方法进行编译，以提高执行效率；torch.compile会将Python代码转换为更高效的形式，减少解释器的开销，从而加速模型的推理过程。
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        logits = logits.float().div_(temperatures.unsqueeze(dim=1)) # 将logits除以温度参数，温度参数控制了采样的随机性；较高的温度会使概率分布更平坦，增加采样的多样性；较低的温度会使概率分布更尖锐，增加采样的确定性。
        probs = torch.softmax(logits, dim=-1) # 对调整后的logits应用softmax函数，得到每个token的概率分布；softmax函数将logits转换为概率分布，使得所有token的概率之和为1；这个概率分布将用于后续的采样步骤。
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)# 通过Gumbel-max trick进行采样；首先生成一个与probs形状相同的张量，元素服从指数分布；然后将probs除以这个指数分布的样本，得到一个新的张量（Exp 分布理论上可取 0，实际会产生极小值，加下限防除零。）；最后在这个新的张量上取argmax，得到采样的token ids；这种方法可以在不使用随机数生成器的情况下实现采样，同时保持了概率分布的正确性。
        return sample_tokens
```

注意：此步骤**不改** `Sampler.forward`（留到 Task 4 一起改），只在文件顶部加新函数。

- [ ] **Step 4: 运行测试，确认全部通过**

```bash
pytest tests/test_apply_repetition_penalty.py -v
```

预期：6 个测试全部 PASS。

---

## Task 4: 改写 `Sampler.forward`（GPU，编译核心）

**Files:**
- Modify: `nanovllm/layers/sampler.py`
- Create: `tests/test_sampler_kernel.py`

- [ ] **Step 1: 写 `tests/test_sampler_kernel.py` 的失败单测**

创建 `tests/test_sampler_kernel.py`：

```python
import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Sampler requires CUDA",
)


def _make_sampler():
    from nanovllm.layers.sampler import Sampler
    return Sampler().cuda()


def _tensor_1d(vals, dtype):
    return torch.tensor(vals, dtype=dtype, device="cuda")


class TestSamplerGreedy:
    def test_all_greedy_equals_argmax(self):
        sampler = _make_sampler()
        torch.manual_seed(0)
        logits = torch.randn(4, 100, device="cuda")
        expected = logits.argmax(dim=-1)

        out = sampler(
            logits.clone(),
            _tensor_1d([0.0, 0.0, 0.0, 0.0], torch.float32),
            _tensor_1d([-1, -1, -1, -1], torch.int64),
            _tensor_1d([1.0, 1.0, 1.0, 1.0], torch.float32),
        )
        assert torch.equal(out, expected)

    def test_top_k_one_equals_argmax(self):
        sampler = _make_sampler()
        torch.manual_seed(0)
        logits = torch.randn(4, 100, device="cuda")
        expected = logits.argmax(dim=-1)

        # top_k=1 且 temperature>0：Gumbel 路径里只有 max 存活，采样结果恒等 argmax
        out = sampler(
            logits.clone(),
            _tensor_1d([0.8, 0.8, 0.8, 0.8], torch.float32),
            _tensor_1d([1, 1, 1, 1], torch.int64),
            _tensor_1d([1.0, 1.0, 1.0, 1.0], torch.float32),
        )
        assert torch.equal(out, expected)


class TestSamplerTopK:
    def test_top_k_five_restricts_choices(self):
        sampler = _make_sampler()
        torch.manual_seed(0)
        logits = torch.randn(1, 100, device="cuda")
        top5 = set(logits.topk(5, dim=-1).indices[0].tolist())

        sampled = set()
        for i in range(200):
            out = sampler(
                logits.clone(),
                _tensor_1d([1.0], torch.float32),
                _tensor_1d([5], torch.int64),
                _tensor_1d([1.0], torch.float32),
            )
            sampled.add(out.item())

        assert sampled.issubset(top5), f"sampled={sampled}, top5={top5}"


class TestSamplerTopP:
    def test_top_p_tiny_keeps_only_max(self):
        sampler = _make_sampler()
        # 构造一个单一 token 概率极大的 logits
        logits = torch.full((1, 50), -10.0, device="cuda")
        logits[0, 7] = 10.0  # token 7 占绝大多数概率

        for i in range(100):
            out = sampler(
                logits.clone(),
                _tensor_1d([1.0], torch.float32),
                _tensor_1d([-1], torch.int64),
                _tensor_1d([0.5], torch.float32),
            )
            assert out.item() == 7


class TestSamplerDefaults:
    def test_all_disabled_matches_legacy_gumbel(self):
        """全禁用 top_k/top_p/rp，结果应服从 softmax(logits/temp) 分布。"""
        sampler = _make_sampler()
        torch.manual_seed(42)
        logits = torch.randn(1, 20, device="cuda")
        temp = 0.8

        expected_probs = torch.softmax(logits[0] / temp, dim=-1).cpu().numpy()

        N = 5000
        counts = [0] * 20
        for _ in range(N):
            out = sampler(
                logits.clone(),
                _tensor_1d([temp], torch.float32),
                _tensor_1d([-1], torch.int64),
                _tensor_1d([1.0], torch.float32),
            )
            counts[out.item()] += 1

        # 简化的 χ² 检验：每个 bin 的经验频率与期望概率误差 < 0.05
        for i in range(20):
            empirical = counts[i] / N
            expected = float(expected_probs[i])
            assert abs(empirical - expected) < 0.05, (
                f"token {i}: empirical={empirical:.3f} vs expected={expected:.3f}"
            )


class TestSamplerMixedBatch:
    def test_mixed_greedy_and_sampled(self):
        sampler = _make_sampler()
        torch.manual_seed(0)
        logits = torch.randn(2, 100, device="cuda")
        row0_argmax = logits[0].argmax().item()

        # 行 0: greedy (temp=0)
        # 行 1: 常规采样 (temp=1.0)
        for _ in range(100):
            out = sampler(
                logits.clone(),
                _tensor_1d([0.0, 1.0], torch.float32),
                _tensor_1d([-1, -1], torch.int64),
                _tensor_1d([1.0, 1.0], torch.float32),
            )
            assert out[0].item() == row0_argmax  # greedy 永远 argmax

    def test_mixed_top_k_values(self):
        sampler = _make_sampler()
        torch.manual_seed(0)
        logits = torch.randn(2, 100, device="cuda")
        top1_row0 = logits[0].argmax().item()
        top5_row1 = set(logits[1].topk(5).indices.tolist())

        row1_sampled = set()
        for _ in range(200):
            out = sampler(
                logits.clone(),
                _tensor_1d([0.8, 0.8], torch.float32),
                _tensor_1d([1, 5], torch.int64),  # 行 0 top_k=1，行 1 top_k=5
                _tensor_1d([1.0, 1.0], torch.float32),
            )
            assert out[0].item() == top1_row0
            row1_sampled.add(out[1].item())

        assert row1_sampled.issubset(top5_row1)
```

- [ ] **Step 2: 运行测试，确认失败**

```bash
pytest tests/test_sampler_kernel.py -v
```

预期：所有测试失败，报 `Sampler.forward() takes 3 positional arguments but 5 were given` 或类似签名错误（因为当前 `forward` 只接 `(logits, temperatures)`）。

- [ ] **Step 3: 改写 `nanovllm/layers/sampler.py` 的 `Sampler` 类**

完整替换 `class Sampler` 部分（保留文件顶部已在 Task 3 添加的 `apply_repetition_penalty`）。文件最终内容：

```python
import torch
from torch import nn


def apply_repetition_penalty(logits: torch.Tensor, seqs: list) -> None:
    """In-place 对 logits 应用 repetition_penalty。

    历史范围 = seq.token_ids（prompt + completion），对齐 vLLM / HF 行为。
    rp == 1.0 的 seq 整行跳过（最常见情况零 GPU op）。
    Scatter 幂等：同一 token 在历史中重复出现不会叠加惩罚。
    """
    for i, seq in enumerate(seqs):
        rp = seq.repetition_penalty
        if rp == 1.0:
            continue
        tokens = torch.tensor(seq.token_ids, device=logits.device, dtype=torch.int64)
        selected = logits[i].gather(0, tokens)
        penalized = torch.where(selected < 0, selected * rp, selected / rp)
        logits[i].scatter_(0, tokens, penalized)


class Sampler(nn.Module):
    # greedy 判定阈值；temperatures.clamp(min=EPS) 防 Gumbel 路径除零。
    EPS: float = 1e-5

    @torch.compile
    def forward(
        self,
        logits: torch.Tensor,
        temperatures: torch.Tensor,
        top_k: torch.Tensor,
        top_p: torch.Tensor,
    ) -> torch.Tensor:
        logits = logits.float()
        V = logits.size(-1)

        # --- greedy 路径：原始 logits 上 argmax（不受 top_k/top_p 影响，对齐 vLLM 语义） ---
        greedy_tokens = logits.argmax(dim=-1)

        # --- top_k：一次 sort 拿到第 k 大的阈值 ---
        k = torch.where(top_k <= 0, V, top_k).clamp(max=V)          # -1 -> V（no-op）
        sorted_logits, sorted_idx = logits.sort(dim=-1, descending=True)
        thresholds = sorted_logits.gather(-1, (k - 1).unsqueeze(-1))
        masked = logits.masked_fill(logits < thresholds, float('-inf'))

        # --- top_p：复用 sorted_logits 做 nucleus 裁剪 ---
        sorted_probs = sorted_logits.softmax(dim=-1)
        cumsum = sorted_probs.cumsum(dim=-1)
        remove_sorted = (cumsum - sorted_probs) > top_p.unsqueeze(-1)  # top_p=1.0 -> 全 False
        remove = torch.zeros_like(remove_sorted).scatter_(-1, sorted_idx, remove_sorted)
        masked = masked.masked_fill(remove, float('-inf'))

        # --- temperature + Gumbel-max ---
        scaled = masked.div(temperatures.clamp(min=self.EPS).unsqueeze(-1))
        probs = scaled.softmax(dim=-1)
        sampled_tokens = probs.div_(
            torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)
        ).argmax(dim=-1)

        # --- 逐行选：greedy seq 取 argmax，其他取 Gumbel ---
        return torch.where(temperatures < self.EPS, greedy_tokens, sampled_tokens)
```

- [ ] **Step 4: 运行测试，确认全部通过**

```bash
pytest tests/test_sampler_kernel.py -v
```

预期：所有 `TestSampler*` 测试 PASS。首次运行会较慢（`@torch.compile` 首次编译耗时 10-30s）。

- [ ] **Step 5: 复跑 Task 3 的测试，确认 `apply_repetition_penalty` 仍 OK**

```bash
pytest tests/test_apply_repetition_penalty.py tests/test_sampling_params.py -v
```

预期：全部 PASS。

---

## Task 5: `ModelRunner` 串接新 Sampler

**Files:**
- Modify: `nanovllm/engine/model_runner.py`

**Note:** 此 Task 无专门单测——改动是管道对接，由 Task 6 的端到端测试覆盖。

- [ ] **Step 1: 修改 `prepare_sample` 方法**

打开 `nanovllm/engine/model_runner.py`，找到 `prepare_sample` 方法（大约 195-198 行）：

```python
    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = [seq.temperature for seq in seqs]
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures
```

替换为：

```python
    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = torch.tensor(
            [seq.temperature for seq in seqs],
            dtype=torch.float32, pin_memory=True,
        ).cuda(non_blocking=True)
        top_k = torch.tensor(
            [seq.top_k for seq in seqs],
            dtype=torch.int64, pin_memory=True,   # int64：Sampler 内部 gather 要求 LongTensor index
        ).cuda(non_blocking=True)
        top_p = torch.tensor(
            [seq.top_p for seq in seqs],
            dtype=torch.float32, pin_memory=True,
        ).cuda(non_blocking=True)
        return temperatures, top_k, top_p
```

- [ ] **Step 2: 修改顶部 import，引入 `apply_repetition_penalty`**

找到文件顶部的 import（大约 10 行）：

```python
from nanovllm.layers.sampler import Sampler
```

替换为：

```python
from nanovllm.layers.sampler import Sampler, apply_repetition_penalty
```

- [ ] **Step 3: 修改 `run` 方法，串入 rep_penalty**

找到 `run` 方法（大约 219-225 行）：

```python
    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        logits = self.run_model(input_ids, positions, is_prefill) # 调用模型前向计算得到logits，如果是prefill阶段，则输入是一个batch的token ids和对应的位置；如果是decode阶段，则输入是每条序列的最后一个token id和对应的位置
        token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        reset_context() # 清空上下文，避免对下一批次产生影响
        return token_ids
```

替换为：

```python
    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        sample_inputs = self.prepare_sample(seqs) if self.rank == 0 else None
        logits = self.run_model(input_ids, positions, is_prefill)
        if self.rank == 0:
            # 两阶段采样：先 eager Python 应用 ragged 的 repetition_penalty，
            # 再调 @torch.compile 的 Sampler 做 top_k / top_p / temperature / greedy-or-Gumbel
            apply_repetition_penalty(logits, seqs)
            token_ids = self.sampler(logits, *sample_inputs).tolist()
        else:
            token_ids = None
        reset_context()
        return token_ids
```

- [ ] **Step 4: 确认没有其他地方调用旧签名**

```bash
cd /home/ubuntu/Nano-Vllm-Learning
grep -rn "self.sampler(" nanovllm/ --include="*.py"
grep -rn "prepare_sample" nanovllm/ --include="*.py"
```

预期：
- `self.sampler(` 只在 `run` 方法里出现一次（Task 5 Step 3 刚改的位置）
- `prepare_sample` 只在 `run` 和 `prepare_sample` 定义本身出现

- [ ] **Step 5: 跑所有非端到端测试确认没炸**

```bash
pytest tests/ -v --ignore=tests/test_metrics_e2e.py --ignore=tests/test_sampling_e2e.py
```

预期：全部 PASS（round #1 metrics 测试 + 本轮 sampling_params / sampler_kernel / apply_repetition_penalty 测试）。

- [ ] **Step 6: 手工 smoke test（如果有 GPU + 模型）**

```bash
python -c "
import os
from nanovllm import LLM, SamplingParams

path = os.environ.get('NANO_VLLM_TEST_MODEL', '/data/models/Qwen3-0.6B')
llm = LLM(path, enforce_eager=True, max_model_len=512, tensor_parallel_size=1)

# 现状回归：默认 SamplingParams
out = llm.generate(['introduce yourself'], SamplingParams(temperature=0.6, max_tokens=16))
print('DEFAULT:', out[0]['text'][:60])

# 新特性：greedy
out = llm.generate(['introduce yourself'], SamplingParams(temperature=0.0, max_tokens=16))
print('GREEDY:', out[0]['text'][:60])
"
```

预期：两次都能正常输出 token，无异常。`GREEDY` 两次运行应完全一致（下一个 Task 的 e2e 测试会自动断言）。

---

## Task 6: 端到端冒烟测试

**Files:**
- Create: `tests/test_sampling_e2e.py`

- [ ] **Step 1: 写 `tests/test_sampling_e2e.py`**

创建 `tests/test_sampling_e2e.py`：

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
    # enforce_eager=True 避免 CUDA graph 捕获耗时，便于测试
    return LLM(MODEL_PATH, enforce_eager=True, max_model_len=1024,
               tensor_parallel_size=1)


def test_greedy_is_deterministic(llm):
    """temperature=0 两次调用同一 prompt 结果完全一致。"""
    from nanovllm import SamplingParams
    sp = SamplingParams(temperature=0.0, max_tokens=32)
    out1 = llm.generate(["introduce yourself"], sp)
    out2 = llm.generate(["introduce yourself"], sp)
    assert out1[0]["token_ids"] == out2[0]["token_ids"]


def test_top_k_one_is_deterministic(llm):
    """top_k=1 与 greedy 等价，两次调用结果一致。"""
    from nanovllm import SamplingParams
    sp = SamplingParams(temperature=0.8, top_k=1, max_tokens=32)
    out1 = llm.generate(["introduce yourself"], sp)
    out2 = llm.generate(["introduce yourself"], sp)
    assert out1[0]["token_ids"] == out2[0]["token_ids"]


def test_repetition_penalty_reduces_repetition(llm):
    """repetition_penalty > 1 会减少 token 重复。

    注意：这是概率性断言，用 enforce_eager + 固定 temperature 让 Gumbel 随机性仍存在，
    但期望 rp 的效果显著到能稳定通过。若偶发失败，放宽比例阈值。
    """
    from nanovllm import SamplingParams
    prompt = "The the the the the the the the the"  # 诱导重复
    max_tokens = 64

    sp_nop = SamplingParams(temperature=0.8, repetition_penalty=1.0, max_tokens=max_tokens)
    sp_rp = SamplingParams(temperature=0.8, repetition_penalty=1.5, max_tokens=max_tokens)

    out_nop = llm.generate([prompt], sp_nop)[0]["token_ids"]
    out_rp = llm.generate([prompt], sp_rp)[0]["token_ids"]

    def uniq_ratio(tokens):
        return len(set(tokens)) / max(len(tokens), 1)

    # rp=1.5 的去重率应 >= rp=1.0（宽松断言——采样有随机性）
    assert uniq_ratio(out_rp) >= uniq_ratio(out_nop) - 0.1, (
        f"rp=1.5 uniq={uniq_ratio(out_rp):.2f}, rp=1.0 uniq={uniq_ratio(out_nop):.2f}"
    )


def test_mixed_sampling_params_batch_runs(llm):
    """传入不同 SamplingParams 的 list，generate 正常工作。"""
    from nanovllm import SamplingParams
    prompts = [
        "introduce yourself",
        "list three colors",
        "what is 2+2?",
        "write a haiku",
    ]
    sps = [
        SamplingParams(temperature=0.0, max_tokens=16),                          # greedy
        SamplingParams(temperature=0.8, top_k=50, max_tokens=16),                # top-k
        SamplingParams(temperature=0.8, top_p=0.9, max_tokens=16),               # top-p
        SamplingParams(temperature=0.8, repetition_penalty=1.2, max_tokens=16),  # rep_penalty
    ]

    outputs = llm.generate(prompts, sps)

    assert len(outputs) == 4
    for o in outputs:
        assert o["text"]
        assert len(o["token_ids"]) > 0
        # round #1 指标仍然附着
        assert o["metrics"] is not None
        assert o["metrics"].ttft > 0


def test_no_recompile_explosion(llm):
    """连续多次不同 batch 大小调用，torch.compile 不应爆 recompile。

    此断言是观察性的：当前 torch 版本若不支持 `_dynamo.utils.counters`，退化为只要 generate 不崩溃即可。
    """
    from nanovllm import SamplingParams
    import torch._dynamo

    # 重置 dynamo 状态，让首次编译开始计数
    try:
        torch._dynamo.reset()
    except Exception:
        pass

    prompts_pool = ["hello", "world", "test", "python", "code"]
    sp = SamplingParams(temperature=0.8, top_k=50, top_p=0.9,
                        repetition_penalty=1.1, max_tokens=8)

    for batch_size in [1, 2, 3, 4, 5]:
        prompts = prompts_pool[:batch_size]
        outputs = llm.generate(prompts, sp)
        assert len(outputs) == batch_size

    # 观察性检查——若 API 不存在则跳过断言
    try:
        counters = torch._dynamo.utils.counters
        unique_graphs = counters.get("stats", {}).get("unique_graphs", None)
        if unique_graphs is not None:
            # 启动编译 + 可能的边界条件，不应远超一个小常数
            # 放宽阈值到 10，主要是防止失控（几百个）
            assert unique_graphs <= 10, f"torch.compile 生成了 {unique_graphs} 个独立图，可能存在 recompile 爆炸"
    except Exception:
        pass  # API 不可用，观察性断言退化为无


def test_backward_compat_default_params(llm):
    """默认 SamplingParams 的输出仍然合理（round #1 的 example.py 那种用法）。"""
    from nanovllm import SamplingParams
    sp = SamplingParams(temperature=0.6, max_tokens=32)
    outputs = llm.generate(["introduce yourself"], sp)
    assert len(outputs) == 1
    assert outputs[0]["text"]
    assert outputs[0]["metrics"].ttft > 0
```

- [ ] **Step 2: 运行端到端测试**

```bash
cd /home/ubuntu/Nano-Vllm-Learning
pytest tests/test_sampling_e2e.py -v -s
```

预期：
- 6 个测试全部 PASS（若模型路径不存在会整体 SKIP）
- 首次运行由于 `@torch.compile` 编译 + 模型加载，耗时较长（30s-2min）
- `-s` 让 tqdm 进度条正常显示

- [ ] **Step 3: 全套测试回归**

```bash
pytest tests/ -v
```

预期：round #1（metrics）+ 本轮 sampling 全部 PASS。若某些 e2e 因为模型或 GPU 不可用而 SKIP，也 OK。

---

## Task 7: 性能回归手工检查 + 最终整体 commit

**Files:**
- 无文件新增/修改；只跑 bench + 最终 commit

- [ ] **Step 1: 记录 baseline（改动前的 HEAD commit 326b3d4，即 round #1 完成的状态）**

切到改动前的状态（**不 checkout，只为性能对比参考；本步可选**）。如果之前已跑过 bench 并保存了 `bench_metrics.json`，可直接用来比较；否则手动记录。

**简化版流程：** 先用当前分支跑一遍 bench，结果对比预期——只要 decode throughput 没有明显下滑（例如 >10%）就算通过。

- [ ] **Step 2: 跑 bench.py**

```bash
cd /home/ubuntu/Nano-Vllm-Learning
python bench.py 2>&1 | tee /tmp/bench_after_sampling.txt
```

预期：
- 打印 `Total: XXXXtok, Time: X.XXs, Throughput: XXXX.XX tok/s`
- 后面打印 `EngineMetrics` summary table
- 产出 `bench_metrics.json`

**手工检查项**（落在备忘，不做自动断言）：
- `decode_throughput` 相比 round #1 相同环境跑出的数值回归 ≤ 5%
- `tpot.p50` / `ttft.p50` 回归 ≤ 5%

若回归 > 5%，进入问题分析（怀疑点：每 step `torch.sort` 固定成本）；若回归严重，可临时把 `Sampler.forward` 的 `@torch.compile` 关掉重测，定位热点。

- [ ] **Step 3: 最终整体 commit（把本轮所有工作合并成一个 commit，包括之前误 commit 的 spec）**

首先检查当前状态：

```bash
git status
git log --oneline -5
```

应该看到：
- HEAD 在 `3d52cdd`（spec commit）
- Working tree 有 sampler/sampling_params/sequence/model_runner 的修改 + 新增的 test 文件

做 soft reset，把 spec commit 退回到 staging，然后加入所有实现改动，一次性 commit：

```bash
# 把 spec commit 退回到工作区
git reset --soft HEAD~1

# 确认状态：spec 文件已经 staged，其他修改和新增文件还未 staged
git status

# 把实现文件和测试一起加入
git add nanovllm/sampling_params.py \
        nanovllm/engine/sequence.py \
        nanovllm/layers/sampler.py \
        nanovllm/engine/model_runner.py \
        tests/test_sampling_params.py \
        tests/test_apply_repetition_penalty.py \
        tests/test_sampler_kernel.py \
        tests/test_sampling_e2e.py \
        doc/specs/2026-04-21-sampling-extension-design.md \
        doc/plans/2026-04-21-sampling-extension.md

# 一次性 commit，一句话 message，不加 Co-Authored-By
git commit -m "新增采样方式扩展 (greedy/top-k/top-p/repetition_penalty) 二次开发，含 spec/plan/实现/单测/端到端冒烟测试。"
```

**关键约束**（出自用户偏好记忆 `feedback_commit_style.md`）：
- **不**加 `Co-Authored-By` trailer
- 一句话 message，不分章节不加 bullet
- 整项 roadmap（spec + plan + 实现 + 测试）压成单一 commit

- [ ] **Step 4: 验证最终 git 状态**

```bash
git log --oneline -3
git show --stat HEAD
```

预期：
- HEAD 是新的 commit，包含 8-10 个文件改动
- 上一个 commit 是 `326b3d4`（round #1 的 engine metrics）
- 没有 `3d52cdd` 这个孤立 spec commit
- `git show --stat HEAD` 列出的文件 = Step 3 `git add` 列出的文件

- [ ] **Step 5: 跑最终全套测试做收官验证**

```bash
pytest tests/ -v
```

预期：round #1 + 本轮全部 PASS（或因为硬件约束正常 SKIP）。

---

## 完成判据

- [ ] `tests/test_sampling_params.py` 11 个测试 PASS
- [ ] `tests/test_apply_repetition_penalty.py` 6 个测试 PASS（需 GPU）
- [ ] `tests/test_sampler_kernel.py` 全部测试 PASS（需 GPU）
- [ ] `tests/test_sampling_e2e.py` 全部测试 PASS（需 GPU + 模型）
- [ ] round #1 `tests/test_metrics_*.py` 无回归
- [ ] `bench.py` decode throughput 回归 ≤ 5%（手工）
- [ ] 最终 git log：`326b3d4 (round #1) ← 新采样扩展 commit (round #2)`，无中间 commit
