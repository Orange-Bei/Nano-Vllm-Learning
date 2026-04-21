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
