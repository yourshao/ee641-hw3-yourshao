"""
Positional encoding implementations for length extrapolation analysis.
"""

import torch
import torch.nn as nn
import math


class SinusoidalPositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding from 'Attention is All You Need'.

    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    This encoding can extrapolate to sequence lengths beyond training.
    """

    def __init__(self, d_model, max_len=5000):
        """
        Initialize sinusoidal positional encoding.

        Args:
            d_model: Embedding dimension
            max_len: Maximum sequence length for precomputation
        """
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        # Create positional encoding matrix [max_len, d_model]
        position = torch.arange(0, max_len).unsqueeze(1)                    # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             (-math.log(10000.0) / d_model))                  # [d_model/2]
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)                      # even indices
        pe[:, 1::2] = torch.cos(position * div_term)                        # odd indices

        # Register as buffer (non-trainable)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Add positional encoding to input.

        Args:
            x: Input tensor [batch, seq_len, d_model]

        Returns:
            x + positional encoding
        """
        seq_len = x.size(1)

        if seq_len <= self.max_len:
            pe = self.pe[:seq_len]
        else:
            # Compute sinusoidal values on-the-fly for longer sequences
            position = torch.arange(0, seq_len, device=x.device).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, self.d_model, 2, device=x.device) *
                                 (-math.log(10000.0) / self.d_model))
            pe = torch.zeros(seq_len, self.d_model, device=x.device)
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)

        return x + pe.unsqueeze(0)  # [1, seq_len, d_model]


class LearnedPositionalEncoding(nn.Module):
    """
    Learned absolute positional embeddings.

    Each position gets a learnable embedding vector.
    Cannot extrapolate beyond max_len seen during training.
    """

    def __init__(self, d_model, max_len=5000):
        """
        Initialize learned positional embeddings.

        Args:
            d_model: Embedding dimension
            max_len: Maximum sequence length
        """
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len

        self.position_embeddings = nn.Embedding(max_len, d_model)

        # Initialize embeddings (normal distribution)
        nn.init.normal_(self.position_embeddings.weight, mean=0.0, std=0.02)

    def forward(self, x):
        """
        Add learned positional embeddings to input.

        Args:
            x: Input tensor [batch, seq_len, d_model]

        Returns:
            x + positional embeddings
        """
        batch_size, seq_len, _ = x.size()

        # Get position indices
        positions = torch.arange(seq_len, device=x.device)
        positions = torch.clamp(positions, max=self.max_len - 1)

        # Look up embeddings and add
        pos_embed = self.position_embeddings(positions)  # [seq_len, d_model]
        return x + pos_embed.unsqueeze(0)  # [1, seq_len, d_model]


class NoPositionalEncoding(nn.Module):
    """
    Baseline: No positional encoding.

    Model is permutation-invariant without position information.
    Should fail on position-dependent tasks like sorting detection.
    """

    def __init__(self, d_model, max_len=5000):
        """
        Initialize no-op positional encoding.

        Args:
            d_model: Embedding dimension (unused)
            max_len: Maximum sequence length (unused)
        """
        super().__init__()
        self.d_model = d_model

    def forward(self, x):
        """
        Return input unchanged.

        Args:
            x: Input tensor [batch, seq_len, d_model]

        Returns:
            x unchanged
        """
        return x


def get_positional_encoding(encoding_type, d_model, max_len=5000):
    """
    Factory function for positional encoding modules.

    Args:
        encoding_type: One of 'sinusoidal', 'learned', 'none'
        d_model: Model dimension
        max_len: Maximum sequence length

    Returns:
        Positional encoding module
    """
    encodings = {
        'sinusoidal': SinusoidalPositionalEncoding,
        'learned': LearnedPositionalEncoding,
        'none': NoPositionalEncoding
    }

    if encoding_type not in encodings:
        raise ValueError(f"Unknown encoding type: {encoding_type}")

    return encodings[encoding_type](d_model, max_len)


def visualize_positional_encoding(encoding_module, max_len=128, d_model=128):
    """
    Visualize positional encoding patterns.

    Args:
        encoding_module: Positional encoding module
        max_len: Number of positions to visualize
        d_model: Model dimension

    Returns:
        Encoding matrix [max_len, d_model] for visualization
    """
    # Create dummy input
    dummy_input = torch.zeros(1, max_len, d_model)

    # Get encoding
    with torch.no_grad():
        encoded = encoding_module(dummy_input)
        encoding = encoded - dummy_input  # Extract just the positional component

    return encoding.squeeze(0).numpy()