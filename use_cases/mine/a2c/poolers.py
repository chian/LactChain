"""Poolers for the last hidden states of a transformer model."""

from __future__ import annotations

from typing import Protocol

import torch


def average_pool(
    embeddings: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Average pool the hidden states using the attention mask.

    Parameters
    ----------
    embeddings : torch.Tensor
        The hidden states to pool (B, SeqLen, HiddenDim).
    attention_mask : torch.Tensor
        The attention mask for the hidden states (B, SeqLen).

    Returns
    -------
    torch.Tensor
        The pooled embeddings (B, HiddenDim).
    """
    # Get the sequence lengths
    seq_lengths = attention_mask.sum(axis=1)
    # Set the attention mask to 0 for start and end tokens
    attention_mask[:, 0] = 0
    attention_mask[:, seq_lengths - 1] = 0

    # Create a mask for the pooling operation (B, SeqLen, HiddenDim)
    pool_mask = attention_mask.unsqueeze(-1).expand(embeddings.shape)
    # Sum the embeddings over the sequence length (use the mask to avoid
    # pad, start, and stop tokens)
    sum_embeds = torch.sum(embeddings * pool_mask, 1)
    # Avoid division by zero for zero length sequences by clamping
    # sum_mask = torch.clamp(pool_mask.sum(1), min=1e-9)
    seq_lengths = torch.clamp(seq_lengths, min=1).unsqueeze(-1)  # Shape (B, 1) to broadcast
    # Compute mean pooled embeddings for each sequence
    return sum_embeds / seq_lengths


def last_token_pool(
    embeddings: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Pool the last hidden states using the attention mask.

    Parameters
    ----------
    embeddings : torch.Tensor
        The last hidden states to pool (B, SeqLen, HiddenDim).
    attention_mask : torch.Tensor
        The attention mask for the hidden states (B, SeqLen).

    Returns
    -------
    torch.Tensor
        The pooled embeddings (B, HiddenDim).
    """
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return embeddings[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = embeddings.shape[0]
        return embeddings[
            torch.arange(batch_size, device=embeddings.device),
            sequence_lengths,
        ]


class Pooler(Protocol):
    """Protocol for pooler functions."""

    def __call__(
        self,
        embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Pool the hidden states using the attention mask."""
        ...


POOLERS: dict[str, Pooler] = {
    'mean': average_pool,
    'last_token': last_token_pool,
}


def get_pooler(pooler_name: str) -> Pooler:
    """Get the pooler function by name.

    Parameters
    ----------
    pooler_name : str
        The name of the pooler function.

    Returns
    -------
    Pooler
        The pooler function.

    Raises
    ------
    ValueError
        If the pooler name is invalid.
    """
    if pooler_name not in POOLERS:
        raise ValueError(f'Invalid pooler name: {pooler_name}')

    return POOLERS[pooler_name]