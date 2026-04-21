<p align="center">
<img width="300" src="assets/logo.png">
</p>

<p align="center">
<a href="https://trendshift.io/repositories/15323" target="_blank"><img src="https://trendshift.io/api/badge/repositories/15323" alt="GeeeekExplorer%2Fnano-vllm | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
</p>

# Nano-vLLM Learning — 二次开发总览

以原 Nano-vLLM 为基础、作为系统性学习二次开发项目，已完成 4 项独立改动。每项都走了完整的 `spec → plan → 实现 → 单测 → 端到端冒烟 → bench sanity` 循环，文档分别在 `doc/specs/`、`doc/plans/`、`doc/reports/`。

| # | 改动 | Commit | 核心成果 | 详细文档 |
|---|---|---|---|---|
| 1 | **引擎指标系统** | [`326b3d4`](../../commit/326b3d4) | TTFT / TPOT / ITL / E2E latency / queue / prefill / decode 7 类 percentile 指标 + KV 利用率 + step-level 采样 | [spec](doc/specs/2026-04-21-engine-metrics-design.md) · [plan](doc/plans/2026-04-21-engine-metrics.md) |
| 2 | **采样方式扩展** | [`431e238`](../../commit/431e238) | 从 Gumbel-max 单一路径 → `greedy` / `top_k` / `top_p` / `repetition_penalty` 四种可组合采样 | [spec](doc/specs/2026-04-21-sampling-extension-design.md) · [plan](doc/plans/2026-04-21-sampling-extension.md) |
| 3 | **Chunked prefill 推广** | [`3454456`](../../commit/3454456) | Budget 填充率 83% → **99.6%**；prefill 吞吐 **+14.1%**；TTFT **−10%** | [spec](doc/specs/2026-04-21-chunked-prefill-generalization-design.md) · [plan](doc/plans/2026-04-21-chunked-prefill-generalization.md) · [A/B report](doc/reports/2026-04-21-chunked-prefill-ab-comparison.md) |
| 4 | **数据并行 DP** | [`d7d796d`](../../commit/d7d796d) | DP=2 总吞吐 **1.58×**；decode 近线性 **1.81×**；时间 **−37%** | [spec](doc/specs/2026-04-21-data-parallel-design.md) · [plan](doc/plans/2026-04-21-data-parallel.md) · [scaling report](doc/reports/2026-04-21-data-parallel-scaling.md) |

### 成果可视化

**改动 #3 · Chunked prefill 推广** (小 budget = 2048, 256 seq 混合长度)

| 指标 | 改前 | 改后 | 变化 |
|---|---|---|---|
| Prefill 吞吐 | 56878 tok/s | **64915 tok/s** | **+14.1%** |
| Prefill 步数 | 84 | **70** | −16.7% |
| Budget 填充率 | 83.0% | **99.6%** | +16.6pp |
| 打满 budget 的步数 | **0 / 84** | **69 / 70** | 从"永远留白"变"几乎填满" |
| TTFT avg / p50 / p90 | 1.50 / 1.53 / 2.30 s | 1.35 / 1.35 / 2.01 s | **−10 ~ −12%** |

> 核心改动：删除 `scheduler.py` 里 3 行的 "first-seq only" chunk 限制——`min(num_tokens, remaining)` 本就是 chunk 实现，移除不必要的 break 即让任意 seq 都能被 chunk 填满 budget。

**改动 #4 · 数据并行 DP** (256 seq，output 100~1024 随机，2×4090)

| 指标 | DP=1 | DP=2 | 变化 |
|---|---|---|---|
| Wall clock throughput | 5272 tok/s | **8329 tok/s** | **1.58×** |
| 总时间 | 25.41 s | **16.08 s** | −37% |
| Decode 吞吐（聚合） | 5626 tok/s | **10166 tok/s** | **1.81×** (近线性) |
| Prefill 吞吐（聚合） | 91799 tok/s | 91162 tok/s | ~持平（亚线性） |
| TTFT avg | 1.025 s | 1.710 s | +67%（sync 节拍代价） |
| 抢占数 | 0 | 0 | — |

> 架构：多进程 DPLLMEngine + round-robin 请求分发 + sync step 节拍 + 聚合 metrics；TP=1 固定不与 TP 组合。Decode 阶段近线性 scaling；prefill 阶段单 worker step_duration 放大是本次未解决的主要瓶颈（CUDA graph warmup 覆盖不全 + PCIe 争用 + Python 开销放大）。详细分析见 scaling report。

### Roadmap

- [x] #1 引擎指标系统 (2026-04-21)
- [x] #2 采样方式扩展 (2026-04-21)
- [x] #3 Chunked prefill 推广 (2026-04-21)
- [x] #4 数据并行 DP (2026-04-21)
- [ ] #5 KV offload（候选，推迟）— 在 Qwen3-0.6B + 4090 场景下 `preemptions=0`、KV 未紧张，触发条件稀缺 A/B 效果难呈现，待更大模型 / 更长 context 场景再开独立 spec

---

# Nano-vLLM

A lightweight vLLM implementation built from scratch.

## Key Features

* 🚀 **Fast offline inference** - Comparable inference speeds to vLLM
* 📖 **Readable codebase** - Clean implementation in ~ 1,200 lines of Python code
* ⚡ **Optimization Suite** - Prefix caching, Tensor Parallelism, Torch compilation, CUDA graph, etc.

## Installation

```bash
pip install git+https://github.com/GeeeekExplorer/nano-vllm.git
```

## Model Download

To download the model weights manually, use the following command:
```bash
huggingface-cli download --resume-download Qwen/Qwen3-0.6B \
  --local-dir ~/huggingface/Qwen3-0.6B/ \
  --local-dir-use-symlinks False
```

## Quick Start

See `example.py` for usage. The API mirrors vLLM's interface with minor differences in the `LLM.generate` method:
```python
from nanovllm import LLM, SamplingParams
llm = LLM("/YOUR/MODEL/PATH", enforce_eager=True, tensor_parallel_size=1)
sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
prompts = ["Hello, Nano-vLLM."]
outputs = llm.generate(prompts, sampling_params)
outputs[0]["text"]
```

## Benchmark

See `bench.py` for benchmark.

**Test Configuration:**
- Hardware: RTX 4070 Laptop (8GB)
- Model: Qwen3-0.6B
- Total Requests: 256 sequences
- Input Length: Randomly sampled between 100–1024 tokens
- Output Length: Randomly sampled between 100–1024 tokens

**Performance Results:**
| Inference Engine | Output Tokens | Time (s) | Throughput (tokens/s) |
|----------------|-------------|----------|-----------------------|
| vLLM           | 133,966     | 98.37    | 1361.84               |
| Nano-vLLM      | 133,966     | 93.41    | 1434.13               |


## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=GeeeekExplorer/nano-vllm&type=Date)](https://www.star-history.com/#GeeeekExplorer/nano-vllm&Date)