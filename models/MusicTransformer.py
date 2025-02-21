import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class RelativeAttentionHead(nn.Module):
    """
    Single head of self-attention with relative positional embeddings (RPE) and causal masking
    """
    def __init__(self, n_embd, dropout, max_len, head_size, mask=True):
        super().__init__()
        self.head_size = head_size
        self.n_embd = n_embd
        self.max_len = max_len
        self.mask = mask
        
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)

        self.rel_emb = nn.Embedding(2 * max_len - 1, head_size)
        
    def forward(self, x):
        B, L, _ = x.size()
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        
        scores = torch.bmm(Q, K.transpose(1, 2)) / (self.head_size ** 0.5)
        M = torch.matmul(Q, self.rel_emb.weight.transpose(0, 1))
        S_rel = self.skewing_trick(M)
        scores = scores + S_rel
        
        if self.mask:
            causal_mask = torch.tril(torch.ones(L, L, device=x.device)).unsqueeze(0)
            scores = scores.masked_fill(causal_mask == 0, float('-inf'))
            
        wei = F.softmax(scores, dim=-1)
        wei_drop = self.dropout(wei)
        out = torch.bmm(wei_drop, V)
        
        return out
        

    def skewing_trick(self, M):
        B, L, _ = M.size()
        M_padded = F.pad(M, (1, 0))
        M_reshaped = M_padded.view(B, -1, L)
        M_shifted = M_reshaped[:, 1:, :]
        S_rel = M_shifted.view(B, L, 2 * self.max_len - 1)[:, :, :L]
        return S_rel

class MultiHeadRelativeAttention(nn.Module):
    ''' multiple relative self attention heads'''
    
    def __init__(self, n_head, n_embd, dropout, max_len, mask=True):
        super().__init__()
        head_size = n_embd // n_head
        self.heads = nn.ModuleList([RelativeAttentionHead(n_embd, dropout, max_len, head_size, mask) for _ in range(n_head)])
        self.proj = nn.Linear(head_size * n_head, n_embd)
        self.dropout = nn.Dropout(dropout)
        self.mask = mask

    def forward(self, x):
        head_outputs = [h(x) for h in self.heads]
        out = torch.cat(head_outputs, dim=-1)
        out = self.dropout(self.proj(out))
        return out
        
class DecoderBlock(nn.Module):
    ''' Decoder block using relative attention '''
    def __init__(self, n_embd, n_head, dropout, max_len, mask=True):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = MultiHeadRelativeAttention(n_head, n_embd, dropout, max_len, mask)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ffwd = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
        
class MusicTransformer(nn.Module):
    ''' 
    Stack decoder blocks for full MusicTransformer class 
    '''

    def __init__(self, vocab_size, n_embd, n_head, n_layer, dropout, max_len):
        super().__init__()

        self.token_emb = nn.Embedding(vocab_size, n_embd) #note: might not need this because we already have relative pos emb
        self.pos_emb = nn.Parameter(torch.zeros(1, max_len, n_embd))
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([DecoderBlock(n_embd, n_head, dropout, max_len) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm = nn.Linear(n_embd, vocab_size)
        
    def forward(self, x):
        B, T = x.shape

        tok_emb = self.token_emb(x)
        pos_emb = self.pos_emb[:, :T, :] 

        x = tok_emb + pos_emb
        x = self.dropout(x)

        for block in self.blocks:
            x = block(x)

        x= self.ln_f(x)
        logits = self.lm(x)

        return logits
