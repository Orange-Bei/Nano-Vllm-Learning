"""Data Parallel 引擎。

N 个 worker 子进程，每进程独占 1 卡跑 LLMEngine(TP=1)。
主进程做 round-robin 请求分发 + sync step 节拍 + 聚合 metrics。
"""
import atexit
import os
from dataclasses import fields
from time import perf_counter

import torch.multiprocessing as mp
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from nanovllm.config import Config
from nanovllm.engine.metrics import EngineMetrics, MetricsCollector


def _dp_worker_entry(rank: int, config_kwargs: dict, child_conn):
    """DP worker 进程入口。必须先设 env var 再 import torch 相关模块。"""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)
    os.environ["NANO_VLLM_DIST_PORT"] = str(2333 + rank)

    from time import perf_counter as _pc
    from nanovllm.engine.llm_engine import LLMEngine
    from nanovllm.engine.sequence import Sequence

    kwargs = {**config_kwargs, "tensor_parallel_size": 1, "data_parallel_size": 1}
    engine = LLMEngine(**kwargs)

    try:
        while True:
            msg = child_conn.recv()
            method = msg[0]
            args = msg[1:]

            if method == "exit":
                child_conn.send(None)
                break

            elif method == "add":
                gid, token_ids, sp = args
                seq = Sequence(token_ids, sp)
                seq.seq_id = gid
                seq.arrival_time = _pc()
                engine.scheduler.add(seq)
                child_conn.send(None)

            elif method == "step":
                # 空 queue 时 scheduler.schedule() 会 assert 失败，这里先判定
                if engine.scheduler.is_finished():
                    child_conn.send(([], 0, False))
                    continue
                t0 = _pc()
                seqs, is_prefill = engine.scheduler.schedule()
                num_batched = sum(s.num_scheduled_tokens for s in seqs) if is_prefill else len(seqs)
                num_tokens = num_batched if is_prefill else -len(seqs)
                token_ids = engine.model_runner.call("run", seqs, is_prefill)
                engine.scheduler.postprocess(seqs, token_ids, is_prefill)
                t1 = _pc()
                engine.metrics.record_step(
                    t0, t1, seqs, is_prefill, engine.scheduler.block_manager, num_batched,
                )
                outputs = [(s.seq_id, s.completion_token_ids) for s in seqs if s.is_finished]
                child_conn.send((outputs, num_tokens, is_prefill))

            elif method == "is_finished":
                child_conn.send(engine.scheduler.is_finished())

            elif method == "metrics_snapshot":
                m = engine.metrics
                child_conn.send({
                    "step_samples": list(m.step_samples),
                    "completed_requests": dict(m.completed_requests),
                    "total_preemptions": m.total_preemptions,
                })

            elif method == "reset_metrics":
                engine.reset_metrics()
                child_conn.send(None)

            else:
                child_conn.send(("error", f"unknown method: {method}"))
    finally:
        engine.exit()


class DPLLMEngine:
    def __init__(self, model, **kwargs):
        config_field_names = {f.name for f in fields(Config)}
        config_kwargs_dict = {k: v for k, v in kwargs.items() if k in config_field_names}
        config_kwargs_dict["model"] = model
        config = Config(**config_kwargs_dict)
        self.data_parallel_size = config.data_parallel_size
        assert self.data_parallel_size > 1, "DPLLMEngine 仅用于 data_parallel_size > 1"

        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        config_kwargs_dict["eos"] = config.eos
        config_kwargs_dict.pop("hf_config", None)

        ctx = mp.get_context("spawn")
        self.ps = []
        self.pipes = []
        for rank in range(self.data_parallel_size):
            parent_conn, child_conn = ctx.Pipe(duplex=True)
            p = ctx.Process(
                target=_dp_worker_entry,
                args=(rank, config_kwargs_dict, child_conn),
            )
            p.start()
            self.ps.append(p)
            self.pipes.append(parent_conn)

        self.next_dispatch_rank = 0
        self.next_global_id = 0
        self._exited = False
        atexit.register(self.exit)

    def add_request(self, prompt, sampling_params):
        if isinstance(prompt, str):
            token_ids = self.tokenizer.encode(prompt)
        else:
            token_ids = prompt
        gid = self.next_global_id
        self.next_global_id += 1
        rank = self.next_dispatch_rank
        self.next_dispatch_rank = (self.next_dispatch_rank + 1) % self.data_parallel_size
        self.pipes[rank].send(("add", gid, token_ids, sampling_params))
        self.pipes[rank].recv()  # ack
        return gid

    def is_finished(self):
        for conn in self.pipes:
            conn.send(("is_finished",))
        # 必须全部 recv（不能用 all() + generator 短路求值，否则剩余 pipe 的响应残留在管道里）
        results = [conn.recv() for conn in self.pipes]
        return all(results)

    def step(self):
        """Broadcast step 给所有 worker，聚合 outputs。"""
        for conn in self.pipes:
            conn.send(("step",))
        all_outputs = []
        total_num_tokens = 0
        for conn in self.pipes:
            outputs, num_tokens, _is_prefill = conn.recv()
            all_outputs.extend(outputs)
            total_num_tokens += num_tokens
        return all_outputs, total_num_tokens

    def generate(self, prompts, sampling_params, use_tqdm=True):
        pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True, disable=not use_tqdm)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)

        gid_order = []
        for prompt, sp in zip(prompts, sampling_params):
            gid = self.add_request(prompt, sp)
            gid_order.append(gid)

        outputs = {}
        prefill_throughput = decode_throughput = 0.
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step()
            if num_tokens > 0:
                prefill_throughput = num_tokens / (perf_counter() - t)
            elif num_tokens < 0:
                decode_throughput = -num_tokens / (perf_counter() - t)
            pbar.set_postfix({
                "Prefill": f"{int(prefill_throughput)}tok/s",
                "Decode": f"{int(decode_throughput)}tok/s",
            })
            for gid, token_ids in output:
                outputs[gid] = token_ids
                pbar.update(1)
        pbar.close()

        for conn in self.pipes:
            conn.send(("metrics_snapshot",))
        merged_reqs = {}
        for conn in self.pipes:
            snap = conn.recv()
            merged_reqs.update(snap["completed_requests"])

        result = []
        for gid in gid_order:
            token_ids = outputs[gid]
            result.append({
                "text": self.tokenizer.decode(token_ids),
                "token_ids": token_ids,
                "metrics": merged_reqs.get(gid),
            })
        return result

    def reset_metrics(self):
        for conn in self.pipes:
            conn.send(("reset_metrics",))
        for conn in self.pipes:
            conn.recv()

    def get_aggregate_metrics(self) -> EngineMetrics:
        merged = MetricsCollector()
        for conn in self.pipes:
            conn.send(("metrics_snapshot",))
        for conn in self.pipes:
            snap = conn.recv()
            merged.step_samples.extend(snap["step_samples"])
            merged.completed_requests.update(snap["completed_requests"])
            merged.total_preemptions += snap["total_preemptions"]
        return merged.build()

    def exit(self):
        if self._exited:
            return
        self._exited = True
        for conn in self.pipes:
            try:
                conn.send(("exit",))
                conn.recv()
            except (BrokenPipeError, EOFError):
                pass
        for p in self.ps:
            p.join(timeout=5)
