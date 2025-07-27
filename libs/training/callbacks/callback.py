from abc import ABC, abstractmethod

from libs.training.pipelines.pipeline import Pipeline


class Callback(ABC):
    @abstractmethod
    def __call__(self, pipeline: Pipeline) -> None: ...
