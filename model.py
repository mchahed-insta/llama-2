import torch 
import torch.nn as nn
import torch.nn.functional as F
import math 
from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32 
    n_heads : int = 32
    n_kv_heads: int Optional[int] = None
    vocab_size: int = -1
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[int] = None
    norm_eps: float = 1e-5 
    # KV cache 
    max_batch_size: int = 32
    max_seq_len: int = 2048 

    device: str = None 

def precompute_theta_pos_frequencies(head_dim: int, seq_len: int, device: str = None, theta: float = 10000.0):
    assert head_dim % 2 == 0
    # Shape (Head_Dim / 2)
    theta_numerator = torch.arange(0, head_dim, 2, device = device).float()
    # Shape (Head_Dim / 2)
    theta = 1.0 / (theta ** (theta_numerator / head_dim)).to(device)
    # Shape (Seq_len)
    m = torch.arange(seq_len, device=device)
    # Shape (Seq_len) outer product (Head_Dim / 2) -> (Seq_len, Head_Dim / 2)
    freqs = torch.outer(m, theta).float()
    #(Seq_len, Head_Dim / 2) -> (Seq_len, Head_Dim)  
    freqs_complex = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_complex


def apply_rotrary_embeddings(x: torch.Tensor, freqs_complex: torch.Tensor, device = str):
    # (B, Seq_len, H, Head_Dim) -> (B, Seq_len, H, Head_Dim / 2, 2)
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1,2))
    # (Seq_len, Head_Dim/2) -> (1, Seq_len, 1, Head_Dim / 2)
    freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(2)
    # (B, Seq_len, H, Head_Dim / 2) * (1, Seq_len, 1, Head_Dim / 2) -> (B, Seq_len, H, Head_Dim / 2)
    x_rotated = x_complex * freqs_complex
    # B, Seq_len, H, Head_Dim / 2, 2) -> (B, Seq_len, H, Head_Dim / 2, 2)
    x_out = torch.view_as_real(x_rotated)
    # (B, Seq_len, H, Head_Dim / 2, 2) -> (B, Seq_len, H, Head_Dim / 2)
    x_out = x_out.reshape(*x.shape)
    return x_out.type_as(x).to(device)


class Transformer(nn.Module):
    
    def __init__(self, args: ModelArgs) -> None:
        super().__init__()
        assert args.vocab_size != -1, "vocab_size must be provided"

        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        self.tok_embeddings = nn.Embedding(self.vocab_size, args.dim)

        self.layers = nn.ModuleList()
        for _ in range(args.n_layers):
            self.layers.append(EncoderBlock(args))

        self.norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.output = nn.Linear(args.dim, self.vocab_size, bias= False)

        self.freqs_complex = precompute_theta_pos_frequencies(self.args.dim // self.args.n_heads, self.args.max_seq_len * 2, device = self.args.device)

    def forward(self, tokens : torch.Tensor, start_pos : int):
        # (B , Seq_len)
        batch_size, seq_len = tokens.shape 
        assert seq_len == 1
        # (B, Seq_len) -> (B, Seq_len, Dim)
        h = self.tok_embeddings(tokens)

        # Retrieve the pairs (m, theta) corresponding to the positions [start_pos, start_pos + seq_len]
        freqs_complex = self.freqs_complex[start_pos: start_pos + seq_len]

        # Consecutively apply all the encoder layers 
        for layer in self.layers:
            h = layer(h, start_pos, freqs_complex)
        h.self.norm(h)
        output = self.output(h).float()
        return output
    




    
         


            

