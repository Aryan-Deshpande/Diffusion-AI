import torch
import torch.nn as nn
from Positional_Embeddings import PE

class Attention(nn.Module):
    def __init__(self, embed_size, seq_len, num_attention_heads):
        super(Attention, self).__init__()
        
        self.d_k = size
        self.qkv_projection =  nn.Linear(embed_size, 3*embed_size)
        self.o_projection = nn.Linear(embed_size, embed_size)
        self.depth = embed_size // num_attention_heads

    def forward(self,x):

        qkv = self.qkv_projection(x)
        q,k,v = torch.chunk(qkv, 3)

        scaled = None

        return attention

ex = Attention(512, 10, 2)
