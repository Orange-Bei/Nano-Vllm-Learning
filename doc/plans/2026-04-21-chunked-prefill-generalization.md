# Chunked Prefill Generalization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 删除 `Scheduler.schedule` prefill 循环里"只允许第一条 seq chunk"的限制，让任意 seq 都能被 chunk，把 `max_num_batched_tokens` 的 budget 填满。

**Architecture:** 改动落在 scheduler 一个文件的 3 行（一个 `if-break` 块）；底层（`model_runner` 的 `prepare_prefill`、flash-attn varlen、block_tables、`postprocess`）已经支持任意 chunk 位置，不动。

**Tech Stack:** Python 3.10+、pytest（已有）、PyTorch（已有）。不新增依赖。

**Spec reference:** `doc/specs/2026-04-21-chunked-prefill-generalization-design.md`

**Commit policy (IMPORTANT):** 本 plan **不做中间 commit**。每个 Task 结束只运行测试、验证通过即可。所有改动累积到 Task 3 最后一步，一次性整体 commit（spec + plan + 实现 + 测试）。本轮 spec 已经按规则没单独 commit，所以不需要 `git reset --soft`。

---

## 文件结构

**新建**

- `tests/test_scheduler_chunked_prefill.py` — 单测，无 GPU，覆盖 scheduler 逻辑
- `tests/test_chunked_prefill_e2e.py` — 端到端活性测试，需 GPU + 模型；用独立的 module 级 `chunked_llm` fixture

**修改**

- `nanovllm/engine/scheduler.py` — 删除 prefill 循环里的 3 行 `if remaining < num_tokens and scheduled_seqs: ... break`

**明确不改**

- 其他所有 `nanovllm/*.py`
- `tests/conftest.py`（session `llm` fixture 保留）
- `tests/test_metrics_e2e.py` / `tests/test_sampling_e2e.py` 等现有 e2e 文件
- `example.py` / `bench.py` / `pyproject.toml`

---

## Task 1: Scheduler 单测 + 应用改动（TDD）

**Files:**
- Create: `tests/test_scheduler_chunked_prefill.py`
- Modify: `nanovllm/engine/scheduler.py`

- [ ] **Step 1: 写 `tests/test_scheduler_chunked_prefill.py` 的单测**

单测构造真实 `Scheduler`（纯 Python，不触 GPU），真实 `BlockManager`、真实 `MetricsCollector`、真实 `Sequence`。注意：
- `Scheduler.__init__` 需要 `Config` 实例 和 `MetricsCollector`
- `Config.__post_init__` 里 `AutoConfig.from_pretrained(self.model)` 会去加载模型配置，为了单测不依赖模型文件，我们不构造 `Config`，而是造一个最简 mock。
- 用一个辅助函数直接构造一个 `Scheduler` 实例（绕过 `__init__` 检查），或使用 `types.SimpleNamespace` 伪造 config。

创建 `tests/test_scheduler_chunked_prefill.py`：

```python
"""Scheduler 的 chunked prefill 推广单测。

不依赖 GPU 或模型文件：直接构造 BlockManager + Scheduler，用真实 Sequence，
手工设置 num_tokens 来模拟不同长度的 prefill 负载。
"""
from types import SimpleNamespace

import pytest

from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.metrics import MetricsCollector
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.sampling_params import SamplingParams


def _make_scheduler(max_num_batched_tokens=8192, max_num_seqs=16,
                    num_kvcache_blocks=64, kvcache_block_size=256, eos=-1):
    """绕过 Config 里的 AutoConfig 加载，直接造 Scheduler 需要的最小接口。"""
    fake_config = SimpleNamespace(
        max_num_batched_tokens=max_num_batched_tokens,
        max_num_seqs=max_num_seqs,
        num_kvcache_blocks=num_kvcache_blocks,
        kvcache_block_size=kvcache_block_size,
        eos=eos,
    )
    metrics = MetricsCollector()
    return Scheduler(fake_config, metrics)


def _make_seq(prompt_len: int):
    """造一个 prompt 长度为 prompt_len 的 Sequence。"""
    return Sequence(list(range(prompt_len)), SamplingParams())


class TestChunkedPrefillScenarios:
    def test_A_all_short_fit_in_one_step(self):
        """场景 A 回归：3 条短 seq 一步全部完成，无 chunk。"""
        sched = _make_scheduler(max_num_batched_tokens=8192)
        for n in (100, 200, 300):
            sched.add(_make_seq(n))

        scheduled, is_prefill = sched.schedule()

        assert is_prefill
        assert len(scheduled) == 3
        # 全部完整入列，没有 chunk
        for seq, expected in zip(scheduled, (100, 200, 300)):
            assert seq.num_scheduled_tokens == expected
            assert seq.status == SequenceStatus.RUNNING
        assert len(sched.waiting) == 0

    def test_B_single_long_seq_gets_chunked(self):
        """场景 B 回归：1 条超长 seq 被 chunk（原来就支持的第一条 seq chunk）。"""
        sched = _make_scheduler(max_num_batched_tokens=512)
        sched.add(_make_seq(2000))  # 远超 budget

        scheduled, is_prefill = sched.schedule()

        assert is_prefill
        assert len(scheduled) == 1
        assert scheduled[0].num_scheduled_tokens == 512   # 被 chunk 到 budget
        assert scheduled[0].status == SequenceStatus.WAITING
        # 没完成，仍在 waiting 队首
        assert len(sched.waiting) == 1
        assert sched.waiting[0] is scheduled[0]

    def test_C_mixed_batch_chunks_last_seq(self):
        """场景 C 新行为：seq0+seq1 完整入列，seq2 被 chunk。

        旧行为：只调度 seq0 和 seq1，seq2 被 break 踢掉（下一步才处理）。
        新行为：seq2 也进本步，被 chunk 到剩余 budget。
        """
        sched = _make_scheduler(max_num_batched_tokens=8192)
        sched.add(_make_seq(3000))  # seq0
        sched.add(_make_seq(5000))  # seq1
        sched.add(_make_seq(2000))  # seq2

        scheduled, is_prefill = sched.schedule()

        assert is_prefill
        assert len(scheduled) == 3  # 新行为：3 条都进本步
        # seq0 和 seq1 完整
        assert scheduled[0].num_scheduled_tokens == 3000
        assert scheduled[0].status == SequenceStatus.RUNNING
        assert scheduled[1].num_scheduled_tokens == 5000
        assert scheduled[1].status == SequenceStatus.RUNNING
        # seq2 被 chunk
        remaining = 8192 - 3000 - 5000   # = 192
        assert scheduled[2].num_scheduled_tokens == remaining
        assert scheduled[2].status == SequenceStatus.WAITING
        # budget 被打满
        total = sum(s.num_scheduled_tokens for s in scheduled)
        assert total == 8192
        # seq2 仍在 waiting 队首等下一步
        assert len(sched.waiting) == 1
        assert sched.waiting[0] is scheduled[2]

    def test_D_chunked_seq_progresses_across_steps(self):
        """场景 D 新行为：被 chunk 的 seq 在后续步骤中继续推进直至完成。"""
        sched = _make_scheduler(max_num_batched_tokens=512)
        seq = _make_seq(1500)  # 需 3 步：512 + 512 + 476
        sched.add(seq)

        # Step 1
        scheduled, _ = sched.schedule()
        assert len(scheduled) == 1
        assert scheduled[0].num_scheduled_tokens == 512
        # 模拟 model_runner 完成这一步：postprocess 推进 num_cached_tokens
        sched.postprocess(scheduled, [0], is_prefill=True)
        assert seq.num_cached_tokens == 512
        assert seq.status == SequenceStatus.WAITING

        # Step 2
        scheduled, _ = sched.schedule()
        assert len(scheduled) == 1
        assert scheduled[0].num_scheduled_tokens == 512
        sched.postprocess(scheduled, [0], is_prefill=True)
        assert seq.num_cached_tokens == 1024

        # Step 3（最后一个 chunk，会生成第一个 token）
        scheduled, _ = sched.schedule()
        assert len(scheduled) == 1
        assert scheduled[0].num_scheduled_tokens == 1500 - 1024  # = 476
        sched.postprocess(scheduled, [42], is_prefill=True)  # 任意 next-token id
        assert seq.num_cached_tokens == 1500 + 1   # prefill 最后一步也 append 一个 token
        assert seq.status == SequenceStatus.RUNNING
        assert seq.last_token == 42

    def test_E_invariants_hold(self):
        """场景 E 不变量：num_batched_tokens 恒 <= max_num_batched_tokens。"""
        sched = _make_scheduler(max_num_batched_tokens=1000, max_num_seqs=8)
        # 造 5 条长度混合的 seq：总和 3700 > 1000
        for n in (400, 600, 800, 500, 1400):
            sched.add(_make_seq(n))

        scheduled, _ = sched.schedule()

        total = sum(s.num_scheduled_tokens for s in scheduled)
        assert total <= 1000
        assert len(scheduled) <= 8
        # 至少把 budget 用了 >= 原始行为（必然有提升或相等）
        # 旧行为：400 + 600 = 1000，然后 break（seq2 800 超 0 剩余）——正好打满
        # 新行为：400 + 600 = 1000，seq2 需 0 剩余也不够——同样结果。选 budget=1001 会让新行为更好
        assert total > 0

    def test_F_chunked_seq_stays_in_waiting_head(self):
        """补充断言：被 chunk 的 seq 在 waiting[0]，下步调度取到同一对象。"""
        sched = _make_scheduler(max_num_batched_tokens=500)
        seq_a = _make_seq(300)
        seq_b = _make_seq(400)   # 300+400=700 > 500，b 会被 chunk 到 200
        sched.add(seq_a)
        sched.add(seq_b)

        scheduled, _ = sched.schedule()
        assert len(scheduled) == 2
        assert scheduled[0].num_scheduled_tokens == 300
        assert scheduled[1].num_scheduled_tokens == 200
        assert scheduled[1] is seq_b
        # a 走完，b 留在 waiting 队首
        assert sched.waiting[0] is seq_b
        assert len(sched.waiting) == 1
```

- [ ] **Step 2: 运行测试，确认 C/D/F 失败（旧行为），A/B/E 通过**

```bash
cd /home/ubuntu/Nano-Vllm-Learning
pytest tests/test_scheduler_chunked_prefill.py -v
```

预期：
- `test_A_all_short_fit_in_one_step` PASS（旧行为支持）
- `test_B_single_long_seq_gets_chunked` PASS（旧行为支持）
- `test_C_mixed_batch_chunks_last_seq` **FAIL**（旧行为 scheduled 只有 2 条）
- `test_D_chunked_seq_progresses_across_steps` PASS（D 中单 seq，旧行为等价支持）—— *注：如果 D 意外失败，说明单 seq chunk 的 postprocess 也有问题，需进一步调查；预期 D 是 PASS，因为旧行为本来就支持单 seq chunk*
- `test_E_invariants_hold` PASS
- `test_F_chunked_seq_stays_in_waiting_head` **FAIL**（旧行为 scheduled 只有 1 条）

- [ ] **Step 3: 应用 scheduler 改动**

打开 `nanovllm/engine/scheduler.py`，找到 `schedule` 方法中 prefill 循环（约 35-57 行）里的这 3 行：

```python
            if remaining < num_tokens and scheduled_seqs:    # 如果剩余的token预算不足以满足该序列的prefill需求，并且已经有其他序列被选中进行prefill了，那么就先不选中该序列，等待下一次调度时再尝试prefill；
                # only allow chunked prefill for the first seq
                break
```

**完整删除这 3 行**（连同注释）。删除后循环应类似（仅展示改动区域）：

```python
        while self.waiting and len(scheduled_seqs) < self.max_num_seqs:
            seq = self.waiting[0]
            num_tokens = max(seq.num_tokens - seq.num_cached_tokens, 1)
            remaining = self.max_num_batched_tokens - num_batched_tokens
            if remaining == 0 or (not seq.block_table and not self.block_manager.can_allocate(seq)):    # no budget
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

- [ ] **Step 4: 重跑单测，确认全部通过**

```bash
pytest tests/test_scheduler_chunked_prefill.py -v
```

预期：全部 6 个测试 PASS。

- [ ] **Step 5: 回归：跑所有非端到端测试**

```bash
pytest tests/ -v --ignore=tests/test_metrics_e2e.py --ignore=tests/test_sampling_e2e.py --ignore=tests/test_chunked_prefill_e2e.py
```

预期：之前 round #1/#2 的 57 个测试 + 本 Task 6 个测试 = 63 个 PASS。

若 round #1 metrics 的单测里有依赖 "first-seq only" 语义的断言（例如 step_samples 的 batch 形状），视具体失败情况决定是不是也要调整。预期没有——metrics 不断言调度策略。

---

## Task 2: E2E 活性测试

**Files:**
- Create: `tests/test_chunked_prefill_e2e.py`

- [ ] **Step 1: 写 `tests/test_chunked_prefill_e2e.py`**

**关键**：不用 `conftest.py` 的 session `llm` fixture，因为需要不同的 `max_num_batched_tokens`；改用本模块自己的 module 级 fixture，并在 teardown 里显式 `exit()` 释放 dist，保证后续模块的 session llm 能正常初始化。

创建 `tests/test_chunked_prefill_e2e.py`：

```python
"""Chunked prefill 推广的端到端活性测试。

独立的 module fixture：`chunked_llm` 用小 max_num_batched_tokens 强制 chunk。
teardown 显式 .exit()，释放 dist 供后续模块使用。
"""
import os
import pytest

MODEL_PATH = os.environ.get("NANO_VLLM_TEST_MODEL", "/data/models/Qwen3-0.6B")
pytestmark = pytest.mark.skipif(
    not os.path.isdir(MODEL_PATH),
    reason=f"Test model not found at {MODEL_PATH}",
)


@pytest.fixture(scope="module")
def chunked_llm():
    from nanovllm import LLM
    instance = LLM(
        MODEL_PATH,
        enforce_eager=True,
        max_model_len=1024,
        max_num_batched_tokens=256,   # 小预算，强制任何较长 prompt 被 chunk
        tensor_parallel_size=1,
    )
    yield instance
    instance.exit()


def test_long_prompt_triggers_chunking(chunked_llm):
    """prompt tokenize 后 >= 256，必然触发 chunked prefill；generate 应正常完成。"""
    from nanovllm import SamplingParams

    # "hello " * 300 大约会 tokenize 到 300 左右的 token（Qwen tokenizer 对常见英文大致 1:1）
    prompt = "hello " * 300
    sp = SamplingParams(temperature=0.0, max_tokens=16)

    outputs = chunked_llm.generate([prompt], sp)

    assert len(outputs) == 1
    o = outputs[0]
    assert o["text"], "completion 不应为空字符串"
    assert len(o["token_ids"]) > 0
    # TTFT 合理
    assert o["metrics"] is not None
    assert o["metrics"].ttft > 0
    # 确认 prompt 确实超了预算
    assert o["metrics"].num_prompt_tokens >= 256, (
        f"prompt 被 tokenize 成 {o['metrics'].num_prompt_tokens} token，没到 256，"
        "e2e 没真正触发 chunking；调大 prompt 重复次数"
    )


def test_chunk_fills_batch_budget(chunked_llm):
    """至少存在一个 prefill step，其 num_batched_tokens == max_num_batched_tokens (256)；
    证明 budget 真的被打满，chunked prefill 生效。
    """
    from nanovllm import SamplingParams

    chunked_llm.reset_metrics()
    prompt = "hello " * 300
    sp = SamplingParams(temperature=0.0, max_tokens=4)   # max_tokens 小，让测试快
    chunked_llm.generate([prompt], sp)

    agg = chunked_llm.get_aggregate_metrics()
    prefill_samples = [s for s in agg.step_samples if s.is_prefill]
    assert prefill_samples, "至少应该有一个 prefill step"

    # 至少有一个 prefill step 打满 256 budget（被 chunk 到极限）
    max_batched = max(s.num_batched_tokens for s in prefill_samples)
    assert max_batched == 256, (
        f"prefill step 的 num_batched_tokens 最大值 = {max_batched}, 预期 256（budget 被打满）。"
        "如果 < 256，说明 chunk 没生效或 prompt 不够长"
    )


def test_mixed_length_batch_completes(chunked_llm):
    """混合长短 prompt 一次性 generate：长的被 chunk、短的正常；全部完成。"""
    from nanovllm import SamplingParams

    prompts = [
        "hello " * 300,   # 超预算，需要 chunk
        "hi",              # 短
        "introduce yourself briefly",   # 中短
    ]
    sp = SamplingParams(temperature=0.0, max_tokens=8)

    outputs = chunked_llm.generate(prompts, sp)

    assert len(outputs) == 3
    for o in outputs:
        assert o["text"], f"输出为空：{o}"
        assert o["metrics"].ttft > 0
```

- [ ] **Step 2: 运行 e2e 测试**

```bash
cd /home/ubuntu/Nano-Vllm-Learning
pytest tests/test_chunked_prefill_e2e.py -v -s
```

预期：3 个测试 PASS（若模型不存在则整体 SKIP）。首次运行因模型加载耗时 30s-1min。`-s` 保留 tqdm 进度条输出。

- [ ] **Step 3: 全套测试回归**

```bash
pytest tests/ -v
```

预期：round #1 + round #2 + 本轮全部 PASS。本轮新增 6（单测）+ 3（e2e）= 9 个测试，总数 63 + 9 = 72 个。

**潜在坑**：若 `test_chunked_prefill_e2e.py` 的 module 级 fixture teardown 没能干净释放 dist，后面 `test_metrics_e2e.py` 或 `test_sampling_e2e.py` 用 conftest 的 session llm 时会崩。本 fixture 已经 `instance.exit()` 处理，配合 round #2 的 `LLMEngine.exit()` 幂等化，应能安全共存。若真出问题，看 `/home/ubuntu/Nano-Vllm-Learning/nanovllm/engine/llm_engine.py` 的 `exit()` 是否走了幂等分支。

---

## Task 3: 性能 sanity + 最终整体 commit

**Files:**
- 无文件新增/修改

- [ ] **Step 1: 跑 bench.py 做 sanity check**

```bash
cd /home/ubuntu/Nano-Vllm-Learning
python bench.py 2>&1 | tail -30
```

预期：和 round #2 末尾跑的数字量级相似或略优（~5000+ tok/s 总吞吐，decode throughput 5000+ tok/s）。若出现明显退步（>10%），怀疑改动引入 regression，需排查。

**非自动化断言**——眼看即可。bench.py 写 `bench_metrics.json` 到工作区，**不要 commit 这个文件**（它是 runtime 产物）。

- [ ] **Step 2: 检查 git 状态**

```bash
git status
git log --oneline -3
```

预期：
- HEAD 在 `431e238`（round #2 sampling commit）
- Working tree 修改：`nanovllm/engine/scheduler.py`
- Working tree 未追踪：`doc/specs/2026-04-21-chunked-prefill-generalization-design.md`、`doc/plans/2026-04-21-chunked-prefill-generalization.md`、`tests/test_scheduler_chunked_prefill.py`、`tests/test_chunked_prefill_e2e.py`、`bench_metrics.json`（不 commit）、`doc/完整demo.md`（用户自己的，不 commit）

- [ ] **Step 3: 整体 commit**

不需要 `git reset --soft`（本轮 spec 没单独 commit）。直接 `git add` 指定文件、单句 message、不加 Co-Authored-By：

```bash
git add \
  nanovllm/engine/scheduler.py \
  tests/test_scheduler_chunked_prefill.py \
  tests/test_chunked_prefill_e2e.py \
  doc/specs/2026-04-21-chunked-prefill-generalization-design.md \
  doc/plans/2026-04-21-chunked-prefill-generalization.md

git commit -m "$(cat <<'EOF'
新增 chunked prefill 推广（任意 seq 可 chunk）二次开发，含 spec/plan/scheduler 单行改动/单测/端到端冒烟测试。
EOF
)"
```

- [ ] **Step 4: 验证最终 git 状态**

```bash
git log --oneline -4
git show --stat HEAD | head -15
```

预期：
- HEAD 新 commit，5 个文件改动（scheduler.py 小幅修改 + 4 个新文件）
- 上一个 commit `431e238`（round #2）
- `bench_metrics.json` 和 `doc/完整demo.md` 仍为未追踪

- [ ] **Step 5: 最后一次全套测试确认**

```bash
pytest tests/ 2>&1 | tail -3
```

预期：72 passed（round #1/2/3 合计）。

---

## 完成判据

- [ ] `tests/test_scheduler_chunked_prefill.py` 6 个单测 PASS
- [ ] `tests/test_chunked_prefill_e2e.py` 3 个端到端测试 PASS（需 GPU + 模型）
- [ ] round #1、#2 所有测试无回归
- [ ] `bench.py` 手工 sanity：吞吐不出现 >10% 退步
- [ ] 最终 git log：`431e238 (round #2) ← 新 commit (round #3)`，无中间 commit
