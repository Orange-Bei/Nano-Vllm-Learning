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
