from torch import nn
from utils import CfgNode
import torch
import math

class NewGELU(nn.Module):
    """ GELU 激活函数 """

    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class CausalSelfAttention(nn.Module):
    """ self attention layer """

    def __init__(self, cfg):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0

class Block(nn.Module):
    """ Transformer block """

    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        # 下面 mlp 不用简单的 nn.Sequential 是因为要在把第二个 Linear 层起名字, 因为它的初始化要特殊处理
        self.mlp = nn.ModuleDict({
            'fc': nn.Linear(cfg.n_embd, 4 * cfg.n_embd),
            'gelu': NewGELU(),
            'proj': nn.Linear(4 * cfg.n_embd, cfg.n_embd),
            'dropout': nn.Dropout(cfg.resid_pdrop)
        })
        m = self.mlp
        self.mlpf = lambda x: m.dropout(m.proj(m.gelu(m.fc(x))))
    
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlpf(self.ln2(x))
        return x


class minGPT(nn.Module):
    """ minGPT """

    @staticmethod   # 静态方法，不需要实例化对象就可以调用
    def get_default_config():
        cfg = CfgNode()
        cfg.n_layer = None
        cfg.n_head = None
        cfg.n_embd = None
        cfg.vocab_size = None
        cfg.block_size = None
        cfg.embd_pdrop = 0.1        # embedding 时的 dropout
        cfg.resid_pdrop = 0.1       # 其余时候的 dropout
        cfg.atten_pdrop = 0.1       # attention 时的 dropout
        return cfg

    def __init__(self, cfg):
        super().__init__()
        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)  # wte: word token embedding
        self.wpe = nn.Embedding(cfg.block_size, cfg.n_embd)  # wpe: word position embedding
        self.embd_pdrop = nn.Dropout(cfg.embd_pdrop)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])   # 这里直接搞 n_layer 层是因为为了初始化参数方便一起弄了
        self.ln = nn.LayerNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)    # 就是最后把 token 映射到词表的那一个矩阵

        self.apply(self._init_weights) # self.apply(func) 初始化一下参数, func 是初始化方式
        # 这里就是对 MLP 里的残差连接权重初始化
        for name, val in self.named_parameters():
            if name.endswith('proj.weight'):
                torch.nn.init.normal_(val, mean=0.0, std=0.02/math.sqrt(2 * cfg.n_layer))

    # 这里直接 copy 的 karpathy 的, 为什么要这么初始化可以去阅读论文
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)


# --------------------- 以下是测试代码 ---------------------
cfg = minGPT.get_default_config()
cfg.n_layer = 3
cfg.n_head = 4
cfg.n_embd = 128
cfg.vocab_size = 1000
cfg.block_size = 128
model = minGPT(cfg)