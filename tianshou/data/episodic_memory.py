"""
Credits to https://github.com/karpathy/minGPT
"""

from dataclasses import dataclass
import numpy as np
from typing import Optional
from typing import Tuple
from einops import rearrange
import torch
import torch.nn as nn
import math
import pandas as pd
from torch.nn import functional as F
from multiprocessing import Queue
from multiprocessing import Process
import gc



torch.backends.cudnn.benchmark=True
torch.backends.cudnn.enabled = True
torch.backends.cudnn.deterministic=True


class Cache:
    def __init__(self, num_samples: int, num_heads: int, max_tokens: int, embed_dim: int, device: torch.device) -> None:
        assert embed_dim % num_heads == 0
        self._n, self._cache, self._size = num_samples, None, None
        self._reset = lambda n: torch.empty(n, num_heads, max_tokens, embed_dim // num_heads, device=device)  # (B, nh, T, hs)
        self.reset()

    @property
    def shape(self) -> Tuple[int, int, int, int]:
        n, num_heads, _, head_dim = self._cache.shape
        return n, num_heads, self._size, head_dim

    def reset(self) -> None:
        self._cache = self._reset(self._n)
        self._size = 0

    def prune(self, mask: np.ndarray) -> None:
        assert mask.ndim == 1 and mask.shape[0] == self.shape[0]
        self._cache = self._cache[mask]
        self._n = self._cache.shape[0]

    def get(self) -> torch.Tensor:
        return self._cache[:, :, :self._size, :]

    def update(self, x: torch.Tensor) -> None:
        assert (x.ndim == self._cache.ndim) and all([x.size(i) == self._cache.size(i) for i in (0, 1, 3)])
        assert self._size + x.size(2) <= self._cache.shape[2]
        self._cache = AssignWithoutInplaceCheck.apply(self._cache, x, 2, self._size, self._size + x.size(2))
        self._size += x.size(2)


class KVCache:
    def __init__(self, n: int, num_heads: int, max_tokens: int, embed_dim: int, device: torch.device) -> None:
        self._k_cache = Cache(n, num_heads, max_tokens, embed_dim, device)
        self._v_cache = Cache(n, num_heads, max_tokens, embed_dim, device)

    @property
    def shape(self) -> Tuple[int, int, int, int]:
        return self._k_cache.shape

    def reset(self) -> None:
        self._k_cache.reset()
        self._v_cache.reset()

    def prune(self, mask: np.ndarray) -> None:
        self._k_cache.prune(mask)
        self._v_cache.prune(mask)

    def get(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._k_cache.get(), self._v_cache.get()

    def update(self, k: torch.Tensor, v: torch.Tensor):
        self._k_cache.update(k)
        self._v_cache.update(v)


class KeysValues:
    def __init__(self, n: int, num_heads: int, max_tokens: int, embed_dim: int, num_layers: int, device: torch.device) -> None:
        self._keys_values = tuple([KVCache(n, num_heads, max_tokens, embed_dim, device) for _ in range(num_layers)])

    def __getitem__(self, key: int) -> KVCache:
        return self._keys_values[key]

    def __len__(self):
        return len(self._keys_values)

    @property
    def size(self):
        return self._keys_values[0].shape[2]

    def reset(self) -> None:
        for kv_cache in self._keys_values:
            kv_cache.reset()

    def prune(self, mask: np.ndarray) -> None:
        for kv_cache in self._keys_values:
            kv_cache.prune(mask)


class AssignWithoutInplaceCheck(torch.autograd.Function):
    """
    Inspired from : https://discuss.pytorch.org/t/disable-in-place-correctness-version-check-any-other-workaround/90738/4
    Warning : do not use it to overwrite a slice twice.
    """

    @staticmethod
    def get_slice(dim: int, start: int, stop: int) -> Tuple[slice]:
        return tuple([slice(None), ] * dim + [slice(start, stop)])

    @staticmethod
    def forward(ctx, input: torch.Tensor, value: torch.Tensor, dim: int, start: int, stop: int) -> torch.Tensor:
        ctx.dim = dim
        ctx.start = start
        ctx.stop = stop
        input.data[AssignWithoutInplaceCheck.get_slice(dim, start, stop)] = value
        return input

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor) -> Tuple[torch.Tensor]:
        return grad_out, grad_out[AssignWithoutInplaceCheck.get_slice(ctx.dim, ctx.start, ctx.stop)], None, None, None

@dataclass
class TransformerConfig:
    tokens_per_block: int
    max_blocks: int
    attention: str

    num_layers: int
    num_heads: int
    embed_dim: int

    embed_pdrop: float
    resid_pdrop: float
    attn_pdrop: float

    @property
    def max_tokens(self):
        return self.tokens_per_block * self.max_blocks


class Transformer(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.drop = nn.Dropout(config.embed_pdrop)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.num_layers)])
        self.ln_f = nn.LayerNorm(config.embed_dim)

    def generate_empty_keys_values(self, n: int, max_tokens: int) -> KeysValues:
        device = self.ln_f.weight.device  # Assumption that all submodules are on the same device
        return KeysValues(n, self.config.num_heads, max_tokens, self.config.embed_dim, self.config.num_layers, device)

    def forward(self, sequences: torch.Tensor, past_keys_values: Optional[KeysValues] = None) -> torch.Tensor:
        assert past_keys_values is None or len(past_keys_values) == len(self.blocks)
        x = self.drop(sequences)
        for i, block in enumerate(self.blocks):
            x = block(x, None if past_keys_values is None else past_keys_values[i])

        x = self.ln_f(x)
        return x


class Block(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(config.embed_dim)
        self.ln2 = nn.LayerNorm(config.embed_dim)
        self.attn = SelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.embed_dim, 4 * config.embed_dim),
            nn.GELU(),
            nn.Linear(4 * config.embed_dim, config.embed_dim),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x: torch.Tensor, past_keys_values: Optional[KeysValues] = None) -> torch.Tensor:
        x_attn, attn = self.attn(self.ln1(x), past_keys_values)
        x = x + x_attn
        x = x + self.mlp(self.ln2(x))
        return x, attn


class SelfAttention(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        assert config.embed_dim % config.num_heads == 0
        assert config.attention in ('causal', 'block_causal')
        self.num_heads = config.num_heads
        self.key = nn.Linear(config.embed_dim, config.embed_dim)
        self.query = nn.Linear(config.embed_dim, config.embed_dim)
        self.value = nn.Linear(config.embed_dim, config.embed_dim)
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        self.proj = nn.Linear(config.embed_dim, config.embed_dim)

        causal_mask = torch.tril(torch.ones(config.max_tokens, config.max_tokens))
        block_causal_mask = torch.max(causal_mask, torch.block_diag(*[torch.ones(config.tokens_per_block, config.tokens_per_block) for _ in range(config.max_blocks)]))
        self.register_buffer('mask', causal_mask if config.attention == 'causal' else block_causal_mask)

    def forward(self, x: torch.Tensor, kv_cache: Optional[KVCache] = None) -> torch.Tensor:
        B, T, C = x.size()
        if kv_cache is not None:
            b, nh, L, c = kv_cache.shape
            assert nh == self.num_heads and b == B and c * nh == C
        else:
            L = 0

        q = self.query(x).view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)   # (B, nh, T, hs)
        k = self.key(x).view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)     # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)   # (B, nh, T, hs)

        if kv_cache is not None:
            kv_cache.update(k, v)
            k, v = kv_cache.get()

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[L:L + T, :L + T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v
        y = rearrange(y, 'b h t e -> b t (h e)')

        y = self.resid_drop(self.proj(y))

        return y, att



############################## here to define the episodic memory network ##################################################

class Memory_Net(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()

        self.config = config
        self.drop = nn.Dropout(config.embed_pdrop)
        self.block = Block(config)
        self.ln_f = nn.LayerNorm(config.embed_dim)

        self.score_net = nn.Sequential(
            nn.Linear(config.embed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)            
        )

    def generate_empty_keys_values(self, n: int, max_tokens: int) -> KeysValues:
        device = self.ln_f.weight.device  # Assumption that all submodules are on the same device
        return KeysValues(n, self.config.num_heads, max_tokens, self.config.embed_dim, self.config.num_layers, device)

    def forward(self, sequences: torch.Tensor, past_keys_values: Optional[KeysValues] = None) -> torch.Tensor:
        assert past_keys_values is None or len(past_keys_values) == len(self.blocks)
        x = self.drop(sequences)
        x, attn = self.block(x, None if past_keys_values is None else past_keys_values[i])

        x = self.ln_f(x)
        x = x[:, -1:, :]
        x = torch.squeeze(x)
        score = self.score_net(x)

        attn = torch.mean(attn, dim = 1)
        attn = attn[:, -1:, :]
        attn = torch.squeeze(attn)

        return score, attn

        
class Memory:

    def __init__(self, args):
        self.args = args
        self.config = TransformerConfig(
            tokens_per_block = self.args['tokens'],
            max_blocks = 1,
            attention = 'causal',
            num_layers = 1,
            num_heads = 8,
            embed_dim = self.args['hidden_size'],
            embed_pdrop = 0.1,
            resid_pdrop = 0.1,
            attn_pdrop = 0.1,
        )

        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.sequence = []
        self.ep_return = []
        self.sequence_len = 0
        self.max_return = 10
        self.min_return = 0
        self.score_list = []
        self.past_returns = []
        self.ema_alpha = 0.1
    
        self.loss_list = []
        self.memory_net = Memory_Net(self.config)
        self.memory_net = nn.DataParallel(self.memory_net)
        self.optimizer = torch.optim.Adam(self.memory_net.parameters(), lr=1e-4)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,step_size=3,gamma = 0.95)

        self.loss_func = nn.MSELoss(reduction='none')
        self.loss = 0

    def update_scores(self, B, C):
        A = np.array([0.0 for i in range(len(B))])
        pos = np.unique(C)
        for p in pos:
            A[np.where(C==p)[0] + p] += B[np.where(C==p)[0]]
        return A

    def compute(self, sequence, ep_score):
        if len(sequence) <= self.config.tokens_per_block:
            self.loss = 0
            self.score_list = [0 for i in range(len(sequence))]
            self.score_list = np.array(self.score_list)
            return 
        tokens_list = np.lib.stride_tricks.sliding_window_view(sequence, (self.config.tokens_per_block, self.config.embed_dim))
        tokens_list = torch.from_numpy(tokens_list.astype(np.float32))
        tokens_list = torch.squeeze(tokens_list)
        scores, attns = self.memory_net(tokens_list)
        del tokens_list

        ep_score = torch.from_numpy(np.array([ep_score]))
        ep_scores = ep_score.expand(np.shape(scores)[0], 1).to(self.device)
        loss = self.loss_func(scores.float(), ep_scores.float())      # mean square error
        del ep_scores

        ################## compute the average score for shaping ###############
        self.past_returns.append(ep_score)
        df = pd.DataFrame(self.past_returns)
        ema = df.ewm(alpha=self.ema_alpha).mean().values 
        avg = ema[-1][0]
        avg_score = (avg - self.min_return) / (self.max_return - self.min_return)

        ################## added by lulu for agrmax attns pos ##################
        
        # scores = torch.squeeze(scores)
        # scores = scores.detach().cpu().numpy()
        # scores = scores - avg_score
        
        # scores = np.pad(scores, (0,self.sequence_len - len(scores)))
        # attns  = attns.detach().cpu().numpy()
        # attn_pos = np.argmax(attns, axis = 1)
        # scores = self.update_scores(scores, attn_pos)         # guide score by attn_pos      
        self.score_list = scores - avg_score
     
        loss = torch.squeeze(loss)
        loss = torch.mean(loss)
        self.loss = loss.item()

        self.loss_list.append(self.loss)

        self.optimizer.zero_grad()               # clear gradients for this training step
        loss.backward()                     # backpropagation, compute gradients
        self.optimizer.step()                    # apply gradients
        self.scheduler.step() 
        
        del loss, attns, scores
        torch.cuda.empty_cache()
        gc.collect()

    def inquire(self):
        return self.score_list * self.args['score_rate']

    def append(self, sequence, rewards):
        self.score_list = []
        self.sequence_len = len(sequence)
        ep_return = sum(rewards)
        self.max_return = max(self.max_return, ep_return)
        self.min_return = min(self.min_return, ep_return)
        ep_score = (ep_return - self.min_return) / (self.max_return - self.min_return)
        self.compute(sequence, ep_score)
        self.sequence = []
        self.ep_return = []

    def kill(self):
        self.alive_pipe.put('kill')

        

