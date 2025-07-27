from typing import Protocol, Iterator, Callable, Dict, Any, Mapping, List, Sequence
from tqdm.auto import tqdm

from libs.schemas.config import LoggingConfig
from libs.schemas.events import MatchableEvent, EventStep, Event
from libs.schemas.train_utils import Phase


class SizedIterable(Protocol):
    def __len__(self) -> int:
        pass

    def __iter__(self) -> Iterator:
        pass


class EventHandler:
    def __init__(self, event: MatchableEvent, function: Callable[..., None], *args, **kwargs) -> None:
        self.event = event
        self.function = function
        self.args: Sequence[Any] = args
        self.kwargs: Dict[str, Any] = kwargs


class OutputHandler:
    def __init__(self, function: Callable[[Any], None]) -> None:
        self.function = function


class Trainer:
    def __init__(
            self,
            process_fn: Callable,
            logging_conf: LoggingConfig,
            show_progress: bool,
            phase: str,
    ):
        self.process_fn = process_fn
        self.phase = phase
        self.show_progress = show_progress
        self.progress_label = "Train" if self.phase == Phase.TRAIN else "Val" if self.phase == Phase.VAL else "Test"
        self.logging_conf = logging_conf

        self.epoch = 0
        self.step = 0
        self.max_epochs = 1

        self.event_handlers: List[EventHandler] = []
        self.output_handlers: List[OutputHandler] = []

    def add_event_handler(
            self,
            event: EventStep | MatchableEvent,
            function: Callable[..., None],
            *args,
            **kwargs
    ) -> None:
        if isinstance(event, str) or isinstance(event, EventStep):
            event = Event(event)
        self.event_handlers.append(EventHandler(event, function, *args, **kwargs))

    def add_output_handler(self, function: Callable[[Any], None]) -> None:
        self.output_handlers.append(OutputHandler(function))

    def run(self, loader: SizedIterable, max_epochs: int = 1) -> None:
        self.max_epochs = max_epochs

        if self._is_done():
            self.epoch = 0
            self.step = 0

        self._fire_event(EventStep.STARTED)

        while not self._is_done():
            self._fire_event(EventStep.EPOCH_STARTED)

            iterations = self._get_data_iterator(loader)
            epoch_length = self._get_data_length(loader)
            progress_bar = self._get_progress_bar(loader)

            for batch_idx in range(epoch_length):
                self._fire_event(EventStep.ITERATION_STARTED)

                batch = next(iterations)
                output = self.process_fn(batch)
                self._handle_output(output)

                self.step += 1
                self._fire_event(EventStep.ITERATION_COMPLETED)
                progress_bar.update()

            progress_bar.close()
            self.epoch += 1
            self._fire_event(EventStep.EPOCH_COMPLETED)

        self._fire_event(EventStep.COMPLETED)

    def state_dict(self) -> Dict[str, Any]:
        return dict(epoch=self.epoch, step=self.step)

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        self.epoch = state_dict["epoch"]
        self.step = state_dict["step"]

    def _fire_event(self, event: EventStep) -> None:
        step = 0
        if event in (EventStep.EPOCH_STARTED, EventStep.EPOCH_COMPLETED):
            step = self.epoch
        if event in (EventStep.ITERATION_STARTED, EventStep.ITERATION_COMPLETED):
            step = self.step

        for handler in self.event_handlers:
            if handler.event.matches(event, step):
                handler.function(*handler.args, **handler.kwargs)

    def _handle_output(self, output: Any) -> None:
        for handler in self.output_handlers:
            handler.function(output)

    def _get_data_iterator(self, loader: SizedIterable) -> Iterator:
        return iter(loader)

    def _get_data_length(self, loader: SizedIterable) -> int:
        return len(loader)

    def _get_progress_label(self) -> str:
        label = ""
        if self.max_epochs > 1:
            label += f"[{self.epoch + 1}/{self.max_epochs}]"
        if self.progress_label is not None and len(self.progress_label) > 0:
            label = self.progress_label + " " + label
        return label

    def _get_progress_bar(self, loader: SizedIterable) -> tqdm:
        desc = self._get_progress_label()
        return tqdm(iterable=loader, desc=desc, ncols=80, leave=False, disable=not self.show_progress)

    def _is_done(self) -> bool:
        return self.epoch >= self.max_epochs
