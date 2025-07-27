from abc import abstractmethod, ABC
from datetime import datetime, timezone
from os import PathLike
from typing import Optional, Sequence, Dict, Any, Mapping, Tuple, Iterator, Callable
import torch
from accelerate import Accelerator
from accelerate.utils import set_seed, LoggerType, TorchDynamoPlugin
from hydra.utils import instantiate
from torch.utils.data import DataLoader

from libs.training.handlers.checkpoint import Checkpoint
from libs.training.helpers.name_generation import generate_name
from libs.schemas.config import Config, ObjectDefinition
from libs.schemas.events import MatchableEvent, EventStep
from libs.schemas.logger import setup_logging
from libs.training.trainers.trainer import SizedIterable
from libs.training.composite_loss import CompositeLoss
from libs.training.trainers.trainer import Trainer


class Pipeline(ABC):
    """Base class for training pipelines."""
    config: Config
    accelerator: Accelerator
    trainer: Trainer

    _run_name: Optional[str] = None

    def __init__(
            self,
            config: Config,
            logging: str = "online",
            wandb_id: Optional[str] = None,
            tags: Optional[Sequence[str]] = None,
            group: Optional[str] = None,
    ):
        """
        Initialize the training pipeline.

        Args:
            config (Config): Configuration object containing all necessary settings for the pipeline.
            logging (str): Logging mode, either "online" or "offline". Defaults to "online".
            wandb_id (Optional[str]): Wandb run ID for online tracking. If None, a new run will be created.
            tags (Optional[Sequence[str]]): Tags to associate with the run.
            group (Optional[str]): Group name for the run, useful for organizing runs in Wandb.
        """
        logging = logging.lower()
        assert logging in ("online", "offline"), f"Invalid logging mode: {logging}. Choose 'online' or 'offline'."

        # Logging
        self.logging_dir = f"{config.logging.dir}/{self.run_name}"
        setup_logging(__name__, out_directory=self.logging_dir)

        # Configuration
        self.config = config
        self.dataset_config = config.dataset
        self.model_config = config.model
        self.pipeline_config = config.pipeline
        self.losses = config.losses
        self.validation_config = config.validation
        self.log_interval = self.validation_config.event
        self.checkpoints_config = config.checkpoints
        self.logging_config = config.logging

        # Set random seed for reproducibility
        if config.pipeline.seed is not None:
            set_seed(config.pipeline.seed)

        # Create accelerator
        self.accelerator = Accelerator(
            log_with=LoggerType.WANDB,  # Todo: add support for Mlflow
            gradient_accumulation_steps=config.pipeline.gradient_accumulation.gradient_accumulation_steps,
            mixed_precision=config.pipeline.amp.type if config.pipeline.amp.enabled else None,
            dynamo_plugin=TorchDynamoPlugin(
                backend=config.pipeline.dynamo.backend,
                mode=config.pipeline.dynamo.mode,
                fullgraph=config.pipeline.dynamo.fullgraph,
                dynamic=config.pipeline.dynamo.dynamic,
                use_regional_compilation=config.pipeline.dynamo.use_regional_compilation,
            ) if config.pipeline.dynamo.enabled else None,
        )

        # Setup Wandb tracking
        self._setup_tracking(mode=logging, wandb_id=wandb_id, tags=tags, group=group)

    @abstractmethod
    def run(self) -> None:
        """Run the training pipeline."""
        ...

    def install_callback(
            self,
            event: EventStep | MatchableEvent,
            callback: Callable[["Pipeline"], None],
            trainer: str = "trainer",
            only_main_process: bool = False,
            **kwargs,
    ) -> None:
        """
        Install callback in the pipeline.

        Parameters
        ----------
        event : MatchableEvent
            Event to trigger callback.
        callback : callable
            Callback function.
            Should take a single argument `pipeline` and return nothing.
        trainer : str
            Which trainer to install callback in. Defaults to "trainer".
        only_main_process : bool
            Install only in the main process. Only affects distributed setups.
        kwargs : keyword arguments
            Optional keyword arguments to be passed to callback.
        """
        if not only_main_process or self.accelerator.is_main_process:
            target = getattr(self, trainer)
            if not isinstance(target, Trainer):
                raise ValueError(f"'{trainer}' is not an trainer. Cannot install callback.")

            target.add_event_handler(event, callback, self, **kwargs)

    def log(self, data: Dict[str, Any]) -> None:
        """Log data to tracker(s)."""
        self.accelerator.log(data, step=self.trainer.step)

    def print(self, *args, **kwargs) -> None:
        """Drop in replacement of `print()` to only print once per server."""
        self.accelerator.print(*args, **kwargs)

    def gather_for_metrics(self, input_data: Any, use_gather_object: bool = False):
        """
        Gathers `input_data` and potentially drops duplicates in the last
        batch if on a distributed system. Should be used for gathering the
        inputs and targets for metric calculation.

        Wrapper around `Accelerator.gather_for_metrics()
        <https://huggingface.co/docs/accelerate/main/en/package_reference/accelerator#accelerate.Accelerator.gather_for_metrics>`_.
        """
        return self.accelerator.gather_for_metrics(input_data, use_gather_object)

    @property
    def device(self) -> torch.device:
        return self.accelerator.device

    @property
    def is_main_process(self) -> bool:
        """True for one process only."""
        return self.accelerator.is_main_process

    @property
    def is_local_main_process(self) -> bool:
        """
        True for one process per server.

        Returns:
            bool: True if the current process is the local main process, False otherwise.
        """
        return self.accelerator.is_local_main_process

    @property
    def run_name(self) -> str:
        """
        Get name of current run.

        Returns:
            str: The name of the current run.
        """
        if self._run_name is None:
            suffix = generate_name()
            now = datetime.now(timezone.utc)
            timestamp = now.strftime("%Y%m%d-%H%M")
            self._run_name = f"{timestamp}-{suffix}"
        return self._run_name

    def _setup_tracking(
            self,
            mode: str,
            wandb_id: Optional[str] = None,
            tags: Optional[Sequence[str]] = None,
            group: Optional[str] = None,
    ) -> None:
        """
        Setup tracking for the pipeline.

        Args:
            mode (str): Mode for tracking, either "online" or "offline".
            wandb_id (Optional[str]): Wandb run ID for online tracking. If None, a new run will be created.
            tags (Optional[Sequence[str]]): Tags to associate with the run.
            group (Optional[str]): Group name for the run, useful for organizing runs in Wandb.

        Raises:
            ValueError: If the mode is not "online" or "offline".
        """
        self.accelerator.init_trackers(
            project_name=self.config.experiment_name,
            config=self.config.model_dump(),
            init_kwargs={
                "wandb": {
                    "id": wandb_id,
                    "mode": mode,
                    "name": self.run_name,
                    "tags": tags,
                    "group": group,
                }
            },
        )

    def _create_composite_loss(self, config: Mapping[str, Any]) -> CompositeLoss:
        """
        Create a composite loss function from the given configuration.

        Args:
            config (Mapping[str, Any]): Configuration for the composite loss function, where keys are loss labels
                and values are dictionaries containing the loss function configuration, including the weight.

        Returns:
            CompositeLoss: An instance of the CompositeLoss class, which combines multiple loss functions.
        """
        loss_labels = []
        loss_modules = []
        loss_weights = []
        for loss_label, loss_conf in config.items():
            cfg_dict = loss_conf.copy()
            weight = cfg_dict.pop("weight", 1.0)

            loss_labels.append(loss_label)
            loss_modules.append(instantiate(cfg_dict))
            loss_weights.append(weight)

        loss_fn = CompositeLoss(labels=loss_labels, losses=loss_modules, weights=loss_weights)
        loss_fn = loss_fn.to(self.device)
        return loss_fn

    def _create_data_loaders(
            self,
            batch_size: int,
            workers: int,
            datasets: Dict[str, ObjectDefinition],
    ) -> Tuple[Dict[str, ObjectDefinition], Dict[str, DataLoader]]:
        """
        Create data loaders for the given datasets.

        Args:
            batch_size (int): Batch size for the data loaders.
            workers (int): Number of worker threads for data loading.
            datasets (Dict[str, ObjectDefinition]): Dictionary of dataset definitions, where keys are split names
                (e.g., "train", "val", "test") and values are the dataset definitions.

        Returns:
            Tuple[Dict[str, ObjectDefinition], Dict[str, DataLoader]]:
                A tuple containing two dictionaries:
                - `out_datasets`: Contains instantiated datasets for each split.
                - `out_loaders`: Contains DataLoader instances for each split.
        """
        out_datasets = {}
        out_loaders = {}

        with self.accelerator.local_main_process_first():
            for split in datasets.keys():
                dataset = instantiate(datasets[split])
                out_datasets[split] = dataset

                out_loaders[split] = DataLoader(
                    dataset=dataset,
                    batch_size=batch_size,
                    num_workers=workers,
                    drop_last=True if split == "train" else False,
                    shuffle=True if split == "train" else False,
                    pin_memory=torch.cuda.is_available(),
                    persistent_workers=True if workers > 0 else False,
                )

        return out_datasets, out_loaders

    def _load_checkpoint(
            self,
            path: str | PathLike,
            to_load: Mapping[str, Any],
            to_unwrap: Optional[Sequence[str]] = None,
            keys: Optional[Sequence[str]] = None,
    ) -> None:
        """
        Load checkpoint from file.

        Args:
            path (str | PathLike): Path to checkpoint file.
            to_load (Mapping[str, Any]): Mapping with objects to load.
            to_unwrap (Optional[Sequence[str]]): Keys for objects to unwrap before loading.
            keys (Optional[Sequence[str]]): List of keys to filter. If None, all keys in `to_load` are used.
        """
        if keys is None:
            keys = list(to_load.keys())
        to_load = {k: to_load[k] for k in keys}

        Checkpoint.load_checkpoint(
            accelerator=self.accelerator,
            path=path,
            to_load=to_load,
            to_unwrap=to_unwrap,
        )
        self.accelerator.wait_for_everyone()

    def _get_data_iterator(self, loader: SizedIterable) -> Iterator:
        """
        Returns an iterator for the given data loader.

        Args:
            loader (SizedIterable): The data loader to iterate over.

        Returns:
            Iterator: An iterator over the data loader.
        """
        return iter(loader)

    def _get_data_length(self, loader: SizedIterable) -> int:
        """
        Returns the length of the data loader.

        Args:
            loader (SizedIterable): The data loader to get the length of.

        Returns:
            int: The length of the data loader.
        """
        return len(loader)
