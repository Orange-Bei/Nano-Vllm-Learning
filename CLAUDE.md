# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Nano-vLLM is a lightweight vLLM implementation (~1,200 lines of Python) supporting offline LLM inference with PagedAttention, prefix caching, tensor parallelism, CUDA graphs, and chunked prefill. Currently supports Qwen3 models only.

## Commands

```bash
# Install
pip install -e .

# Run inference
python example.py

# Run benchmark (256 sequences, random 100-1024 token lengths)
python bench.py
```

There is no test suite, linter, or formatter configured.

## Architecture

### Inference Pipeline

`LLM` (thin subclass of `LLMEngine`) is the user-facing entry point. `generate()` drives a loop:

1. **Scheduler** (`engine/scheduler.py`) picks sequences to run — prefill first (with chunked prefill support), then decode. Manages waiting/running queues and preemption when KV-cache blocks run out.
2. **ModelRunner** (`engine/model_runner.py`) prepares input tensors (input_ids, positions, slot_mapping, block_tables) and runs the model. Uses CUDA graphs for decode batches ≤512 when `enforce_eager=False`.
3. **Sampler** (`layers/sampler.py`) applies temperature-scaled sampling via Gumbel-max trick.

### KV-Cache & Memory

- **BlockManager** (`engine/block_manager.py`) implements paged KV-cache with fixed-size blocks (default 256 tokens). Blocks are content-addressed via xxhash for **prefix caching** — sequences sharing a prompt prefix reuse cached blocks.
- **Sequence** (`engine/sequence.py`) tracks per-request state: token_ids, block_table, scheduling status (WAITING → RUNNING → FINISHED). Has custom `__getstate__`/`__setstate__` for efficient pickling across processes.

### Tensor Parallelism

- Rank 0 is the main process; ranks 1+ are spawned via `multiprocessing` and run `ModelRunner.loop()`.
- Rank 0 sends method calls (serialized via pickle) through POSIX shared memory; worker ranks wait on `multiprocessing.Event`.
- Linear layers split along column (`ColumnParallelLinear`, `QKVParallelLinear`, `MergedColumnParallelLinear`) or row (`RowParallelLinear`) with NCCL all-reduce on row-parallel output.
- Embedding/LM head split vocabulary across ranks (`VocabParallelEmbedding`, `ParallelLMHead`).

### Attention

- `layers/attention.py` uses **flash-attn**: `flash_attn_varlen_func` for prefill, `flash_attn_with_kvcache` for decode.
- A **Triton kernel** (`store_kvcache_kernel`) writes K/V into the paged cache via slot_mapping.
- Attention metadata is passed through a **global Context** (`utils/context.py`) — set before each forward pass, reset after.

### Weight Loading

`utils/loader.py` loads safetensors checkpoints. `Qwen3ForCausalLM.packed_modules_mapping` maps HF weight names to merged parameter names (e.g., `q_proj`/`k_proj`/`v_proj` → `qkv_proj`, `gate_proj`/`up_proj` → `gate_up_proj`). Each parameter has a `weight_loader` callback that handles TP sharding.

### torch.compile Usage

`@torch.compile` is applied to: `Sampler.forward`, `RotaryEmbedding.forward`, `SiluAndMul.forward`, `RMSNorm.rms_forward`, `RMSNorm.add_rms_forward`.
