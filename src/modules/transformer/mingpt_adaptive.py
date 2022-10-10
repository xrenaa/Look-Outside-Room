"""
credit to: https://github.com/karpathy/minGPT/
GPT model:
- the initial stem consists of a combination of token encoding and a positional encoding
- the meat of it is a uniform sequence of Transformer blocks
    - each Transformer is a sequential combination of a 1-hidden-layer MLP block and a self-attention block
    - all blocks feed into a central residual pathway similar to resnets
- the final decoder is a linear projection into a vanilla Softmax classifier
"""

import math
import logging
import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F

from src.main import instantiate_from_config

logger = logging.getLogger(__name__)

def get_sinusoid_encoding(n_position, d_hid):
    ''' Sinusoid position encoding table '''

    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)


class GPTConfig:
    """ base GPT config, params common to all GPT versions """
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1

    def __init__(self, vocab_size, block_size, **kwargs):
        self.vocab_size = vocab_size
        self.block_size = block_size
        for k,v in kwargs.items():
            setattr(self, k, v)


class GPT1Config(GPTConfig):
    """ GPT-1 like network roughly 125M params """
    n_layer = 12
    n_head = 12
    n_embd = 768
    

class AdaptiveAttention(nn.Module):
    def __init__(self, block_size, time_len = 3, camera_dim = 30, img_dim = 256):
        super().__init__()
        self.block_size = block_size
        self.camera_dim = camera_dim
        self.img_dim = img_dim
        
        self.fc = nn.Sequential(
            nn.Linear(camera_dim, 2 * camera_dim),
            nn.GELU(),  # nice
            nn.Linear(2 * camera_dim, img_dim**2),
        )
        
    def forward(self, p1=None, p2=None, p3=None):
        # hand-craft assign:
        B = p1.shape[0]
        h = torch.zeros(B, 1, self.block_size, self.block_size).cuda()
        # C 0->1
        if p1 is not None:
            h_01 = self.fc(p1).view(B, 1, self.img_dim, self.img_dim)
            h[:, :, 285:541, 0:256] = h_01
        # C 0->2
        if p2 is not None:
            h_02 = self.fc(p2).view(B, 1, self.img_dim, self.img_dim)
            h[:, :, 571:827, 0:256] = h_02
        # C 1->2
        if p3 is not None:
            h_12 = self.fc(p3).view(B, 1, self.img_dim, self.img_dim)
            h[:, :, 571:827, 286:542] = h_12

        return h

class CausalSelfAttention(nn.Module):
    def __init__(self, config, adaptive):
        super().__init__()
        assert config.n_embd % config.n_head == 0, f"n_embd is {config.n_embd} but n_head is {config.n_head}."
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        mask = torch.tril(torch.ones(config.block_size,
                                     config.block_size))
        if hasattr(config, "n_unmasked"):
            mask[:config.n_unmasked, :config.n_unmasked] = 1
        self.register_buffer("mask", mask.view(1, 1, config.block_size, config.block_size))
        self.n_head = config.n_head
        
        self.adaptive = adaptive
        
    def forward(self, x, h, layer_past=None):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        
        if self.adaptive:
            att = h[:,:,:T,:T] + att
        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y


class Block(nn.Module):
    """ an unassuming Transformer block """
    def __init__(self, config, adaptive):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config, adaptive)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),  # nice
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x, p):
        x = x + self.attn(self.ln1(x), p)
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    """  the full GPT language model, with a context size of block_size """
    def __init__(self, vocab_size, block_size, time_len, n_layer=12, n_head=8, n_embd=256,
                 embd_pdrop=0., resid_pdrop=0., attn_pdrop=0., n_unmasked=0,
                 input_vocab_size=None):
        super().__init__()
        config = GPTConfig(vocab_size=vocab_size, block_size=block_size,
                           embd_pdrop=embd_pdrop, resid_pdrop=resid_pdrop, attn_pdrop=attn_pdrop,
                           n_layer=n_layer, n_head=n_head, n_embd=n_embd,
                           n_unmasked=n_unmasked)
        # input embedding stem
        in_vocab_size = vocab_size if not input_vocab_size else input_vocab_size
        self.tok_emb = nn.Embedding(in_vocab_size, config.n_embd)
        
        # Locality
        self.locality = AdaptiveAttention(config.block_size)
        
        # init pos embedding
        self.time_len = time_len
        self.frame_emb = nn.Parameter(torch.zeros(1, 256, config.n_embd))
        self.camera_emb = nn.Parameter(torch.zeros(1, 30, config.n_embd))
        self.role_emb = None
        
        self.time_emb = nn.Parameter(data=get_sinusoid_encoding(n_position=block_size, d_hid=config.n_embd), requires_grad=False)
        
        # dropout
        self.drop = nn.Dropout(config.embd_pdrop)
        # transformer 
        self.blocks = nn.ModuleList()
        
        for _ in range(int(config.n_layer // 2)):
            self.blocks.append(Block(config, adaptive = True))
            self.blocks.append(Block(config, adaptive = False))

        # decoder head
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.block_size = config.block_size
        self.apply(self._init_weights)
        self.config = config
        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def iter_forward(self, dc_emb, z_emb, p, embeddings=None, targets=None, return_layers=False):
        
        token_embeddings_dc = dc_emb

        # add the token embedding with z_indices
        token_embeddings_z = z_emb
        token_embeddings = torch.cat([token_embeddings_dc, token_embeddings_z], 1)
        token_embeddings = token_embeddings[:, :-1, :] # remove the last one
        
        # drop out for teacher
        token_embeddings = self.drop(token_embeddings)
        
        t = token_embeddings.shape[1]
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."
        
        role_emb = []
        for _ in range(self.time_len-1):
            role_emb.append(self.frame_emb)
            role_emb.append(self.camera_emb)

        role_emb.append(self.frame_emb)
        role_emb = torch.cat(role_emb, 1)
        
        role_embeddings = role_emb[:, :t, :] # each position maps to a (learnable) vector
        time_embeddings = self.time_emb[:, :t, :] # each position maps to a (learnable) vector
        
        x = token_embeddings + role_embeddings + time_embeddings

        if return_layers:
            layers = [x]
            for block in self.blocks:
                x = block(x)
                layers.append(x)
            return layers

        # locality
        p1, p2, p3 = p
        h = self.locality(p1, p2, p3)
        # h = h.repeat(x.shape[0], 1, 1, 1)
        for block in self.blocks:
            x = block(x, h)
        
        # x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            
        return logits, loss
    
    def test(self, dc_emb, z_indices, p, embeddings=None, targets=None, return_layers=False, return_bias=False):
        token_embeddings_dc = dc_emb

        # add the token embedding with z_indices
        token_embeddings_z = self.tok_emb(z_indices)
        token_embeddings = torch.cat([token_embeddings_dc, token_embeddings_z], 1)
        
        if embeddings is not None:  # prepend explicit embeddings
            token_embeddings = torch.cat((embeddings, token_embeddings), dim=1)

        t = token_embeddings.shape[1]
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."
        
        role_emb = []
        for _ in range(self.time_len-1):
            role_emb.append(self.frame_emb)
            role_emb.append(self.camera_emb)

        role_emb.append(self.frame_emb)
        role_emb = torch.cat(role_emb, 1)
        
        role_embeddings = role_emb[:, :t, :] # each position maps to a (learnable) vector
        time_embeddings = self.time_emb[:, :t, :] # each position maps to a (learnable) vector
        
        x = token_embeddings + role_embeddings + time_embeddings

        # locality
        p1, p2, p3 = p
        h = self.locality(p1, p2, p3)
        h = h.repeat(x.shape[0], 1, 1, 1)
        for block in self.blocks:
            x = block(x, h)
            
        x = self.ln_f(x)
        logits = self.head(x)

        # if we are given some desired targets also calculate the loss
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        if return_bias:
            return logits, h
        else:
            return logits, loss

class DummyGPT(nn.Module):
    # for debugging
    def __init__(self, add_value=1):
        super().__init__()
        self.add_value = add_value

    def forward(self, idx):
        return idx + self.add_value, None

#### sampling utils
def top_k_logits(logits, k):
    v, ix = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[:, [-1]]] = -float('Inf')
    return out

@torch.no_grad()
def sample(model, x, steps, temperature=1.0, sample=False, top_k=None):
    """
    take a conditioning sequence of indices in x (of shape (b,t)) and predict the next token in
    the sequence, feeding the predictions back into the model each time. Clearly the sampling
    has quadratic complexity unlike an RNN that is only linear, and has a finite context window
    of block_size, unlike an RNN that has an infinite context window.
    """
    block_size = model.get_block_size()
    model.eval()
    for k in range(steps):
        x_cond = x if x.size(1) <= block_size else x[:, -block_size:]  # crop context if needed
        logits, _ = model(x_cond)
        # pluck the logits at the final step and scale by temperature
        logits = logits[:, -1, :] / temperature
        # optionally crop probabilities to only the top k options
        if top_k is not None:
            logits = top_k_logits(logits, top_k)
        # apply softmax to convert to probabilities
        probs = F.softmax(logits, dim=-1)
        # sample from the distribution or take the most likely
        if sample:
            ix = torch.multinomial(probs, num_samples=1)
        else:
            _, ix = torch.topk(probs, k=1, dim=-1)
        # append to the sequence and continue
        x = torch.cat((x, ix), dim=1)

    return x