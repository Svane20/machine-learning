import inspect
from typing import Sequence, Optional, Mapping, Callable, Any
from torch import nn, Tensor


class CompositeLoss(nn.Module):
    def __init__(
            self,
            labels: Sequence[str],
            losses: Sequence[nn.Module],
            weights: Sequence[float],
            transforms: Optional[Mapping[str, Callable[[Any, Any], Any]]] = None,
    ):
        super().__init__()

        assert len(labels) == len(losses) == len(weights), "Labels, losses, and weights must have the same length."

        self.labels = labels
        self.losses = nn.ModuleList(losses)
        self.weights = weights
        if transforms is None:
            transforms = {}
        self.transforms = transforms
        self.last_values = [None] * len(losses)

    def _gather_extra_args(self, loss_fn: nn.Module, kwargs) -> Mapping[str, Any]:
        sig = inspect.signature(loss_fn.forward)
        args = sig.parameters.keys()
        return {k: v for k, v in kwargs.items() if k in args}

    def forward(self, input: Tensor, target: Tensor, **kwargs) -> float:
        """
        Compute loss.
        """
        total_loss = 0.0
        for i, (weight, fn, label) in enumerate(zip(self.weights, self.losses, self.labels)):
            args = (input, target)
            if label in self.transforms:
                args = self.transforms[label](*args)
            extra_args = self._gather_extra_args(fn, kwargs)
            loss = weight * fn(*args, **extra_args)
            total_loss += loss
            self.last_values[i] = loss.item()
        return total_loss
