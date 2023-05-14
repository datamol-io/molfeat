from typing import List
from typing import Optional
from typing import Callable

import torch

from torch import nn
from loguru import logger


def get_default_hgf_pooler(name, config, **kwargs):
    """Get default HuggingFace pooler based on the model name
    Args:
        name: name of the model
        config: config of the model
        kwargs: additional arguments to pass to the pooler
    """
    model_type = config.get("model_type", None)
    if name not in ["bert", "roberta", "gpt", "bart"] and name in Pooling.SUPPORTED_POOLING[:-1]:
        return HFPooler(config, name=name, **kwargs)
    names = [name]
    if model_type is not None:
        names += [model_type]
    if any(x in ["bert", "roberta"] for x in names):
        return BertPooler(config, **kwargs)
    elif any(x.startswith("gpt") for x in names):
        return GPTPooler(config, **kwargs)
    elif any(x == "bart" for x in names):
        return BartPooler(config, **kwargs)
    return None


class Pooling(nn.Module):
    """
    Perform simple pooling on a tensor over one dimension
    """

    SUPPORTED_POOLING = ["mean", "avg", "max", "sum", "clf", None]

    def __init__(self, dim: int = 1, name: str = "max"):
        """
        Pooling for embeddings

        Args:
            dim: dimension to pool over, default is 1
            name: pooling type. Default is 'mean'.
        """
        super().__init__()
        self.dim = dim
        self.name = name

    def forward(self, x, indices: List[int] = None, mask: torch.Tensor = None) -> torch.Tensor:
        """Perform a pooling operation on the input tensor

        Args:
            x: input tensor to pull over
            indices: Subset of indices to pool over. Defaults to None for all indices.
            mask: binary mask to apply when pooling. Defaults to None, which is a matrix of 1.
                If mask is provided it takes precedence over indices.
        """
        x = torch.as_tensor(x)
        if mask is None:
            mask = torch.ones_like(x)
        if indices is not None:
            mask[:, indices] = 0
        neg_inf = torch.finfo(x.dtype).min
        if mask.ndim == 2:
            mask = mask.unsqueeze(-1)  # B, S, 1
        if self.name == "clf":
            return x[:, 0, :]
        if self.name == "max":
            tmp = x.masked_fill(mask, neg_inf)
            return torch.max(tmp, dim=self.dim)[0]
        elif self.name in ["mean", "avg"]:
            return torch.sum(x * mask, dim=self.dim) / mask.sum(self.dim)
        elif self.name == "sum":
            return torch.sum(x * mask, dim=self.dim)
        return x


class HFPooler(nn.Module):
    """Default Pooler based on Molfeat Pooling layer"""

    def __init__(self, config, dim: int = 1, name: str = "mean", **kwargs):
        super().__init__()
        self.config = config
        self.pooling = Pooling(dim=dim, name=name)

    def forward(
        self,
        h: torch.Tensor,
        inputs: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        ignore_padding: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass of the pooling layer

        Args:
            h: hidden representation of the input sequence to pool over
            inputs: optional input that has been provided to the underlying bert model
            mask: optional mask to use in place of computing the padding specific mask
            ignore_padding: whether to ignore padding tokens when pooling

        Returns:
            pooled_output: pooled representation of the input sequence
        """
        if mask is None and ignore_padding:
            mask = inputs.ne(self.config.get("pad_token_id"))
        if mask.ndim == 2:
            mask = mask.unsqueeze(-1)  # B, S, 1
        return self.pooling(h, indices=None, mask=mask)


class BertPooler(nn.Module):
    """
    Default Bert pooler as implemented in huggingface transformers
    The bert pooling function focuses on a projection of the first token ([CLS]) to get a sentence representation.
    """

    def __init__(
        self,
        config,
        activation: Optional[Callable] = None,
        random_seed: int = None,
        **kwargs,
    ):
        super().__init__()
        self.config = config
        self.random_seed = random_seed
        if self.random_seed is not None:
            torch.manual_seed(self.random_seed)
        hidden_size = config.get("hidden_size")
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh() if activation is None else activation

    def forward(
        self, h: torch.Tensor, inputs: Optional[torch.Tensor] = None, **kwargs
    ) -> torch.Tensor:
        """Forward pass of the pooling layer

        Args:
            h: hidden representation of the input sequence to pool over
            inputs: optional input that has been provided to the underlying bert model

        Returns:
            pooled_output: pooled representation of the input sequence
        """
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = h[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BartPooler(nn.Module):
    """
    Default Bart pooler as implemented in huggingface transformers
    The Bart pooling function focusing on the eos token ([EOS]) to get a sentence representation.
    """

    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config

    def forward(
        self, h: torch.Tensor, inputs: Optional[torch.Tensor] = None, **kwargs
    ) -> torch.Tensor:
        """Forward pass of the pooling layer

        Args:
            h: hidden representation of the input sequence to pool over
            inputs: inputs tokens to the bart underlying model

        Returns:
            pooled_output: pooled representation of the input sequence
        """
        eos_mask = inputs.eq(self.config.get("eos_token_id"))
        if len(torch.unique_consecutive(eos_mask.sum(1))) > 1:
            raise ValueError("All examples must have the same number of <eos> tokens.")
        pooled_output = h[eos_mask, :].view(h.size(0), -1, h.size(-1))[:, -1, :]
        return pooled_output


class GPTPooler(nn.Module):
    """
    Default GPT pooler as implemented in huggingface transformers
    The GPT pooling function focusing on the last non-padding token given sequence length to get a sentence representation.
    """

    def __init__(self, config, **kwargs):
        super().__init__()
        self.config = config
        self.pad_token_id = config.get("pad_token_id")

    def forward(
        self, h: torch.Tensor, inputs: Optional[torch.Tensor] = None, **kwargs
    ) -> torch.Tensor:
        """Forward pass of the pooling layer

        Args:
            h: hidden representation of the input sequence to pool over
            inputs: inputs tokens to the bart underlying model

        Returns:
            pooled_output: pooled representation of the input sequence
        """
        batch_size, sequence_lengths = inputs.shape[:2]

        assert (
            self.pad_token_id is not None or batch_size == 1
        ), "Cannot handle batch sizes > 1 if no padding token is defined."
        if self.pad_token_id is None:
            sequence_lengths = -1
            logger.warning(
                f"{self.__class__.__name__} will not detect padding tokens in `inputs_embeds`. Results may be "
                f"unexpected if using padding tokens in conjunction with `inputs_embeds.`"
            )
        else:
            sequence_lengths = torch.ne(inputs, self.pad_token_id).sum(-1) - 1
        pooled_output = h[torch.arange(batch_size), sequence_lengths]
        return pooled_output
