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
