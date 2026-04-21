import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm # 进度条库，用于显示生成进度和速度
from transformers import AutoTokenizer
import torch.multiprocessing as mp # 多进程库，用于在多GPU环境下启动多个模型runner进程

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner


class LLMEngine:

    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        Sequence.block_size = config.kvcache_block_size
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn") # 使用spawn方式启动子进程，避免fork带来的问题
        for i in range(1, config.tensor_parallel_size): # 启动tensor parallel的子进程，rank 0的模型推理在主进程中执行
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = ModelRunner(config, 0, self.events) # 主进程也启动一个model runner，负责rank 0的模型推理
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)
        atexit.register(self.exit) # 保证 Python 退出时无论正常结束还是异常都会清理子进程和共享内存。

    def exit(self):
        self.model_runner.call("exit") # 触发子 rank 跳出 loop，退出进程
        del self.model_runner # rank 0 自己 exit
        for p in self.ps:
            p.join() # 等子进程退出

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)

    def step(self):
        seqs, is_prefill = self.scheduler.schedule()
        num_tokens = sum(seq.num_scheduled_tokens for seq in seqs) if is_prefill else -len(seqs) # 如果是prefill阶段，则统计本次总的token数量；如果是decode阶段，则统计本次的序列数量（每条序列decode一个token）
        token_ids = self.model_runner.call("run", seqs, is_prefill) # 调用model runner执行当前的batch，得到生成的token ids
        self.scheduler.postprocess(seqs, token_ids, is_prefill) # 根据生成的token ids更新对应的序列状态，并判断是否结束
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        return outputs, num_tokens

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True, disable=not use_tqdm)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        outputs = {}
        prefill_throughput = decode_throughput = 0.
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step()
            if num_tokens > 0:
                prefill_throughput = num_tokens / (perf_counter() - t)
            else:
                decode_throughput = -num_tokens / (perf_counter() - t)
            pbar.set_postfix({ # 显示当前的prefill和decode速度
                "Prefill": f"{int(prefill_throughput)}tok/s",
                "Decode": f"{int(decode_throughput)}tok/s",
            })
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                pbar.update(1)
        pbar.close()
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())] # 按照请求的顺序返回生成的token ids列表
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs] # 将生成的token ids解码成文本，并返回文本和token ids的列表
        return outputs
