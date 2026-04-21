import json
import os
import time
from random import randint, seed
from nanovllm import LLM, SamplingParams


def main():
    seed(0)
    num_seqs = 256
    max_input_len = 1024
    max_ouput_len = 1024

    path = os.path.expanduser("/data/models/Qwen3-0.6B")
    llm = LLM(path, enforce_eager=False, max_model_len=4096)

    prompt_token_ids = [[randint(0, 10000) for _ in range(randint(100, max_input_len))] for _ in range(num_seqs)]
    sampling_params = [SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=randint(100, max_ouput_len)) for _ in range(num_seqs)]

    # warmup
    llm.generate(["Benchmark: "], SamplingParams())
    llm.reset_metrics()

    t = time.time()
    llm.generate(prompt_token_ids, sampling_params, use_tqdm=False)
    t = time.time() - t

    total_tokens = sum(sp.max_tokens for sp in sampling_params)
    throughput = total_tokens / t
    print(f"Total: {total_tokens}tok, Time: {t:.2f}s, Throughput: {throughput:.2f}tok/s")
    print()

    metrics = llm.get_aggregate_metrics()
    print(metrics.summary_table())

    json_path = "temp/bench-runs/bench_metrics.json"
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(metrics.to_dict(), f, indent=2)
    print(f"\nRaw metrics exported to {json_path}")


if __name__ == "__main__":
    main()
