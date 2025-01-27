from torch import nn
from my_mingpt.utils import CfgNode
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
        assert cfg.n_embd % cfg.n_head == 0, 'embedding 的维度必须是多头的整数倍'
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.attn_dropout = nn.Dropout(cfg.atten_pdrop)
        self.resid_dropout = nn.Dropout(cfg.resid_pdrop)
        # 定义一个不被优化器优化的 buffer, 名字叫 mask, 这个 buffer 会被保存在 state_dict 中
        # tril 是取下三角矩阵, view 是把它变成 4 维的, 为了和后面的 attention 矩阵相乘
        self.register_buffer("mask", torch.tril(torch.ones(cfg.block_size, cfg.block_size)).view(1, 1, cfg.block_size, cfg.block_size))
        self.softmax = nn.Softmax()
        self.n_head = cfg.n_head
        self.n_embd = cfg.n_embd

    def forward(self, x):
        # B 表示 batch, T 表示 token 的数量, C 表示每个 token 的维度
        B, T, C = x.shape
        # (B, T, 3 * C) -> (B, T, C) * 3
        q, k, v = self.qkv(x).split(dim=-1, split_size=C)

        # (B, T, C) -> (B, T, n_head, C // n_head) -> (B, n_head, T, C // n_head)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # att: (B, n_head, T, T)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(C // self.n_head)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        att = nn.Softmax(dim=-1)(att)
        att = self.attn_dropout(att)
        y = att @ v # (B, n_head, T, C // n_head) x (B, n_head, T, C // n_head) -> (B, n_head, T, C // n_head)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # (B, T, n_head, C // n_head) -> (B, T, C)
        return self.resid_dropout(self.proj(y))

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
        self.cfg = cfg
        self.wte = nn.Embedding(cfg.vocab_size, cfg.n_embd)  # wte: word token embedding
        self.wpe = nn.Embedding(cfg.block_size, cfg.n_embd)  # wpe: word position embedding
        self.embd_drop = nn.Dropout(cfg.embd_pdrop)
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

    def forward(self, x, y=None):
        B, T = x.shape # batch, token 数量
        assert T <= self.cfg.block_size, 'token 数量超过了 block_size'
        pos = torch.arange(0, T, dtype=torch.long, device=x.device).unsqueeze(0) # (1, T)

        tok_emb = self.wte(x) # (B, T, C)
        pos_emb = self.wpe(pos) # (1, T, C)
        x = self.embd_drop(tok_emb + pos_emb) # (B, T, C)
        # 广播的规则是，如果某个维度上的大小不一致，且其中一个维度为 1，那么它会被自动扩展以匹配另一个维度
        for block in self.blocks:
            x = block(x)
        x = self.ln(x)
        logits = self.lm_head(x)    # (B, T, vocab_size)

        loss = None
        if y is not None:
            # 将 logits 展为 (BxT, vocab_size), y 展为 (BxT,)
            loss = nn.CrossEntropyLoss(ignore_index=-1)(logits.view(-1, logits.shape[-1]), y.view(-1))

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, do_sample=False, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.shape[1] <= self.cfg.block_size else idx[:, -self.cfg.block_size:]
            logits, _ = self(idx_cond) # logits: (B, T, vocab_size)
            logits = logits[:, -1, :] / temperature # 只保留最后一个 token 的 logits
            if top_k is not None:
                v, _ = torch.topk(logits, top_k, dim=-1)
                logits[logits < v[:, [-1]]] = -float('Inf') # logits: (B, vocab_size)
            probs = nn.Softmax(dim=-1)(logits)
            if do_sample:
                idx_nxt = torch.multinomial(probs, num_samples=1)
            else:
                _, idx_nxt = torch.topk(probs, k=1, dim=-1) # idx_nxt: (B, 1)
            idx = torch.cat((idx, idx_nxt), dim=1)
        return idx