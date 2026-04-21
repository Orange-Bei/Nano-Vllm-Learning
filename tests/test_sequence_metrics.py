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
