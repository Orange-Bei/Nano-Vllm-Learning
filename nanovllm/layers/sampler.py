import torch
from torch import nn


class Sampler(nn.Module):

    @torch.compile # 使用torch.compile装饰器对forward方法进行编译，以提高执行效率；torch.compile会将Python代码转换为更高效的形式，减少解释器的开销，从而加速模型的推理过程。
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        logits = logits.float().div_(temperatures.unsqueeze(dim=1)) # 将logits除以温度参数，温度参数控制了采样的随机性；较高的温度会使概率分布更平坦，增加采样的多样性；较低的温度会使概率分布更尖锐，增加采样的确定性。
        probs = torch.softmax(logits, dim=-1) # 对调整后的logits应用softmax函数，得到每个token的概率分布；softmax函数将logits转换为概率分布，使得所有token的概率之和为1；这个概率分布将用于后续的采样步骤。
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)# 通过Gumbel-max trick进行采样；首先生成一个与probs形状相同的张量，元素服从指数分布；然后将probs除以这个指数分布的样本，得到一个新的张量（Exp 分布理论上可取 0，实际会产生极小值，加下限防除零。）；最后在这个新的张量上取argmax，得到采样的token ids；这种方法可以在不使用随机数生成器的情况下实现采样，同时保持了概率分布的正确性。
        return sample_tokens
