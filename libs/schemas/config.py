from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, Union, Optional, List

from accelerate.utils import PrecisionType, DynamoBackend
from omegaconf import MISSING

from libs.schemas.events import EventStep

ObjectDefinition = Dict[str, Any]
WrappedObjectDefinition = Dict[str, ObjectDefinition]


class ConfigType(str, Enum):
    """Pipeline configuration type."""

    SUPERVISED = "supervised"
    GAN = "gan"


@dataclass
class LogInterval:
    event: Union[EventStep, str] = EventStep.EPOCH_COMPLETED
    every: int = 1
    first: Optional[int] = None
    last: Optional[int] = None

    def model_dump(self) -> ObjectDefinition:
        return {
            "event": self.event,
            "every": self.every,
            "first": self.first,
            "last": self.last
        }


class IntervalStrategy(str, Enum):
    STEPS = "steps"
    EPOCH = "epoch"


@dataclass
class LoggingConfig:
    dir: Optional[str] = None
    strategy: Union[IntervalStrategy, str] = IntervalStrategy.STEPS
    every: int = 500


class SchedulerType(str, Enum):
    """
    Parameter scheduler type.
    """

    LINEAR = "linear"
    COSINE = "cosine"


@dataclass
class SchedulerConfig:
    type: Union[SchedulerType, str] = SchedulerType.COSINE
    end_value: float = 1e-7
    warmup_steps: int = 2

    def model_dump(self) -> ObjectDefinition:
        return {
            "type": self.type,
            "end_value": self.end_value,
            "warmup_steps": self.warmup_steps,
        }


@dataclass
class AMPConfig:
    enabled: bool = True
    type: Optional[Union[PrecisionType, str]] = PrecisionType.FP16

    def model_dump(self) -> ObjectDefinition:
        return {
            "enabled": self.enabled,
            "type": self.type,
        }


@dataclass
class GradientAccumulationConfig:
    gradient_accumulation_steps: int = 1
    clip_grad_norm: Optional[float] = None
    clip_grad_value: Optional[float] = None

    def model_dump(self) -> ObjectDefinition:
        return {
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "clip_grad_norm": self.clip_grad_norm if self.clip_grad_norm is not None else None,
            "clip_grad_value": self.clip_grad_value if self.clip_grad_value is not None else None,
        }


@dataclass
class DynamoConfig:
    enabled: bool = False
    use_regional_compilation: Optional[bool] = False
    backend: Optional[Union[DynamoBackend, str]] = DynamoBackend.INDUCTOR
    mode: Optional[str] = "default"
    fullgraph: Optional[bool] = True
    dynamic: Optional[bool] = False

    def model_dump(self) -> ObjectDefinition:
        return {
            "enabled": self.enabled,
            "backend": self.backend,
            "mode": self.mode,
            "fullgraph": self.fullgraph,
            "dynamic": self.dynamic
        }


@dataclass
class PipelineConfig:
    seed: Optional[int] = None
    gradient_accumulation: GradientAccumulationConfig = field(default_factory=GradientAccumulationConfig)
    amp: AMPConfig = field(default_factory=AMPConfig)
    dynamo: DynamoConfig = field(default_factory=DynamoConfig)
    optimizer: ObjectDefinition = field(default_factory=dict)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)

    def model_dump(self) -> ObjectDefinition:
        return {
            "seed": self.seed,
            "gradient_accumulation": self.gradient_accumulation.model_dump(),
            "amp": self.amp.model_dump(),
            "dynamo": self.dynamo.model_dump(),
            "optimizer": self.optimizer,
            "scheduler": self.scheduler.model_dump()
        }


@dataclass
class GanPipelineConfig(PipelineConfig):
    disc_scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    disc_optimizer: ObjectDefinition = field(default_factory=dict)

    def model_dump(self) -> ObjectDefinition:
        base_dump = super().model_dump()
        return {
            **base_dump,
            "disc_scheduler": self.disc_scheduler.model_dump(),
            "disc_optimizer": self.disc_optimizer
        }


@dataclass
class ValidationConfig:
    event: Union[EventStep | str] = EventStep.EPOCH_COMPLETED
    metrics: Optional[WrappedObjectDefinition] = field(default_factory=dict)

    def model_dump(self) -> ObjectDefinition:
        return {
            "interval": self.event,
            "metrics": self.metrics
        }


class CheckpointMode(str, Enum):
    """Checkpoint evaluation mode."""
    MIN = "min"
    MAX = "max"


@dataclass
class CheckpointConfig:
    metric: Optional[str] = None
    mode: Union[CheckpointMode, str] = CheckpointMode.MAX
    num_saved: int = 1
    interval: LogInterval = field(default_factory=LogInterval)

    def model_dump(self) -> ObjectDefinition:
        return {
            "metric": self.metric,
            "mode": self.mode,
            "num_saved": self.num_saved,
            "interval": self.interval.model_dump(),
        }


@dataclass
class Config:
    type: Union[ConfigType, str] = MISSING
    experiment_name: str = MISSING
    show_progress: bool = True
    max_epochs: int = 50
    batch_size: int = 32
    loader_workers: int = 4

    dataset: WrappedObjectDefinition = field(default_factory=dict)
    model: ObjectDefinition = field(default_factory=dict)
    losses: WrappedObjectDefinition = field(default_factory=dict)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    checkpoints: List[CheckpointConfig] = field(default_factory=list)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    def model_dump(self) -> Any:
        return {
            "type": self.type,
            "experiment_name": self.experiment_name,
            "max_epochs": self.max_epochs,
            "batch_size": self.batch_size,
            "loader_workers": self.loader_workers,
            "dataset": self.dataset,
            "model": self.model,
            "losses": self.losses,
            "validation": self.validation.model_dump(),
            "checkpoints": [checkpoint.model_dump() for checkpoint in self.checkpoints],
        }


@dataclass
class SupervisedConfig(Config):
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)

    def model_dump(self) -> ObjectDefinition:
        base_dump = super().model_dump()
        return {
            **base_dump,
            "pipeline": self.pipeline.model_dump(),
        }


@dataclass
class GANConfig(Config):
    disc_model: ObjectDefinition = field(default_factory=dict)
    pipeline: GanPipelineConfig = field(default_factory=GanPipelineConfig)
    disc_losses: WrappedObjectDefinition = field(default_factory=dict)

    def model_dump(self) -> ObjectDefinition:
        base_dump = super().model_dump()
        return {
            **base_dump,
            "disc_model": self.disc_model,
            "pipeline": self.pipeline.model_dump(),
            "disc_losses": self.disc_losses,
        }
