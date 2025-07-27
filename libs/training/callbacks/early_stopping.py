from typing import Dict, Any

from accelerate import Accelerator


class EarlyStopping:
    def __init__(
            self,
            accelerator: Accelerator,
            patience: int,
            metric_name: str,
            mode: str = "min",
            min_delta: float = 0.0,
            metrics: Dict[str, Any] = None
    ) -> None:
        self.accelerator = accelerator
        self.patience = patience
        self.metric_name = metric_name
        self.mode = mode.lower()
        self.min_delta = min_delta
        self.metrics = metrics if metrics is not None else {}
        self.best = float('inf') if self.mode == "min" else float('-inf')
        self.counter = 0

    def __call__(self):
        if hasattr(self.accelerator, "is_main_process") and not self.accelerator.is_main_process:
            return
        if self.metric_name not in self.metrics:
            return

        metric_obj = self.metrics[self.metric_name]
        try:
            current_val = metric_obj.compute().item()
        except Exception:
            current_val = float(metric_obj.compute())

        improved = False
        if self.mode == "min":
            if current_val < self.best - self.min_delta:
                improved = True
        else:
            if current_val > self.best + self.min_delta:
                improved = True

        if improved:
            self.best = current_val
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.accelerator.print(
                    f"EarlyStopping: '{self.metric_name}' has not improved for {self.patience} evaluations. "
                    f"Stopping training."
                )
                self.accelerator.end_training()
