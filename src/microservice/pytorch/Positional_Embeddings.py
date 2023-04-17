
import math
import torch
import torch.nn as nn

# Sinosudial Positional Encoder

# PE
def PE():
    d_model = 6
    max_sequence_length = 100

    positions = torch.arange(max_sequence_length).reshape(max_sequence_length, 1)
    print(positions[0:3])
    print(positions.shape)

    even = torch.arange(0, d_model, 2).float()
    odd = torch.arange(0, d_model, 1).float()

    even_denom = torch.pow(1000, even/d_model)
    denominator = even_denom

    pe1 = torch.sin(positions/denominator)
    pe2 = torch.cos(positions/denominator)

    t = torch.stack((pe1,pe2),dim=2)
    print(t)
    t = torch.flatten(t)
    print(t)

"""embeddings = math.log(10000) / (half_dim) * embeddings
embeddings = torch.exp(torch.arange(half_dim) * -embeddings)
embeddings = time[:, None] * embeddings[None, :]

embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)

print(embeddings)"""
