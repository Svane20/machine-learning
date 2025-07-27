from typing import Mapping, Callable, Optional, Any
from torchmetrics import Metric

from libs.schemas.events import EventStep
from libs.training.trainers.trainer import Trainer


class MetricLogger:
    def __init__(
            self,
            metrics: Mapping[str, Metric],
            log_function: Callable[[Any], None],
            prefix: Optional[str] = None,
    ):
        self._metrics = metrics
        self._log_function = log_function
        if prefix is None:
            prefix = ""
        self._prefix = prefix

    def attach(self, trainer: Trainer) -> None:
        trainer.add_output_handler(self._handle_output)
        trainer.add_event_handler(event=EventStep.EPOCH_STARTED, function=self._epoch_started)
        trainer.add_event_handler(event=EventStep.EPOCH_COMPLETED, function=self._epoch_completed)

    def _handle_output(self, outputs) -> None:
        for metric in self._metrics.values():
            metric(*outputs)

    def _epoch_started(self) -> None:
        for metric in self._metrics.values():
            metric.reset()

    def _epoch_completed(self) -> None:
        data = {}
        for label, metric in self._metrics.items():
            data[self._prefix + label] = metric.compute().item()
        self._log_function(data)
