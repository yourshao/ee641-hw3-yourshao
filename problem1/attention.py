"""
Attention mechanisms for sequence-to-sequence modeling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Compute scaled dot-product attention.

    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V

    Args:
        Q: Query tensor [batch, ..., seq_len_q, d_k]
        K: Key tensor [batch, ..., seq_len_k, d_k]
        V: Value tensor [batch, ..., seq_len_v, d_k]
        mask: Optional mask [batch, ..., seq_len_q, seq_len_k]
              Values: 1 for positions to attend, 0 for positions to mask

    Returns:
        output: Attention output [batch, ..., seq_len_q, d_k]
        attention_weights: Attention weights [batch, ..., seq_len_q, seq_len_k]
    """
    d_k = Q.size(-1)

    # Compute attention scores: QK^T
    scores = torch.matmul(Q, K.transpose(-2, -1))  # [..., q, k]

    # Scale scores
    scores = scores / math.sqrt(d_k)

    # Apply mask (1=keep, 0=mask)
    if mask is not None:
        # mask should be broadcastable to scores' shape
        scores = scores.masked_fill(mask == 0, float("-inf"))

    # Softmax over key dimension
    attention_weights = F.softmax(scores, dim=-1)

    # Weighted sum of values
    output = torch.matmul(attention_weights, V)  # [..., q, d_k]

    return output, attention_weights


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism.

    Splits d_model into num_heads, applies attention in parallel,
    then concatenates and projects the results.
    """

    def __init__(self, d_model, num_heads):
        """
        Initialize multi-head attention.

        Args:
            d_model: Model dimension (must be divisible by num_heads)
            num_heads: Number of attention heads
        """
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Linear projections for Q, K, V
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)

        # Output projection
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def split_heads(self, x):
        """
        Split tensor into multiple heads.

        Args:
            x: Input tensor [batch, seq_len, d_model]

        Returns:
            Tensor with shape [batch, num_heads, seq_len, d_k]
        """
        batch_size, seq_len, _ = x.size()
        # reshape -> [batch, seq_len, num_heads, d_k] -> transpose to [batch, num_heads, seq_len, d_k]
        x = x.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        return x

    def combine_heads(self, x):
        """
        Combine multiple heads back into single tensor.

        Args:
            x: Input tensor [batch, num_heads, seq_len, d_k]

        Returns:
            Tensor with shape [batch, seq_len, d_model]
        """
        batch_size, _, seq_len, d_k = x.size()
        # transpose -> [batch, seq_len, num_heads, d_k] -> reshape to [batch, seq_len, d_model]
        x = x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return x

    def forward(self, query, key, value, mask=None):
        """
        Forward pass of multi-head attention.

        Args:
            query: Query tensor [batch, seq_len_q, d_model]
            key: Key tensor [batch, seq_len_k, d_model]
            value: Value tensor [batch, seq_len_v, d_model]
            mask: Optional attention mask

        Returns:
            output: Attention output [batch, seq_len_q, d_model]
            attention_weights: Attention weights [batch, num_heads, seq_len_q, seq_len_k]
        """
        batch_size = query.size(0)

        # Linear projections
        Q = self.W_q(query)  # [B, Lq, d_model]
        K = self.W_k(key)    # [B, Lk, d_model]
        V = self.W_v(value)  # [B, Lv, d_model]

        # Split heads
        Q = self.split_heads(Q)  # [B, H, Lq, d_k]
        K = self.split_heads(K)  # [B, H, Lk, d_k]
        V = self.split_heads(V)  # [B, H, Lv, d_k]

        # If mask is provided, make it broadcastable to [B, H, Lq, Lk]
        if mask is not None and mask.dim() == 3:
            # e.g., [B, Lq, Lk] -> [B, 1, Lq, Lk]
            mask = mask.unsqueeze(1)

        # Scaled dot-product attention
        attn_out, attn_weights = scaled_dot_product_attention(Q, K, V, mask=mask)  # [B,H,Lq,d_k], [B,H,Lq,Lk]

        # Combine heads
        out = self.combine_heads(attn_out)  # [B, Lq, d_model]

        # Output projection
        out = self.W_o(out)

        return out, attn_weights


def create_causal_mask(seq_len, device=None):
    """
    Create causal mask to prevent attending to future positions.

    Args:
        seq_len: Sequence length
        device: Device to create tensor on

    Returns:
        Mask tensor [1, 1, seq_len, seq_len] lower triangular matrix
    """
    # Lower triangular matrix
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
    return mask.unsqueeze(0).unsqueeze(0)
