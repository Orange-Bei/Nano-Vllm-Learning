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
