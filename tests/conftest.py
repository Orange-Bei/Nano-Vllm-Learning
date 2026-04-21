"""共享 test fixture：整个 session 只创建一个 LLM，避免重复 dist.init_process_group 冲突。"""
import os
import pytest

MODEL_PATH = os.environ.get("NANO_VLLM_TEST_MODEL", "/data/models/Qwen3-0.6B")


@pytest.fixture(scope="session")
def llm():
    """session 级 LLM 实例，供所有 e2e 测试共享。session 结束时显式 exit 清理 dist。"""
    if not os.path.isdir(MODEL_PATH):
        pytest.skip(f"Test model not found at {MODEL_PATH}")
    from nanovllm import LLM
    instance = LLM(MODEL_PATH, enforce_eager=True, max_model_len=1024,
                   tensor_parallel_size=1)
    yield instance
    instance.exit()
