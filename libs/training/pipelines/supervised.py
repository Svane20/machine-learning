import logging
import os
import warnings
from functools import partial
from os import PathLike
from typing import Optional, Sequence

import numpy as np
import torch
from hydra.utils import instantiate

from libs.training.handlers.checkpoint import Checkpoint
from libs.training.handlers.composite_loss_logger import CompositeLossLogger
from libs.training.handlers.metric_logger import MetricLogger
from libs.training.handlers.optimizer_logger import OptimizerLogger
from libs.training.handlers.output_logger import OutputLogger
from libs.training.helpers.parse_log_interval import parse_log_interval
from libs.schemas.config import SupervisedConfig
from libs.training.pipelines.pipeline import Pipeline
from libs.training.lr_scheduler import create_lr_scheduler
from libs.training.trainers.supervised import SupervisedTrainer, SupervisedEvaluator

class SupervisedPipeline(Pipeline):
    def __init__(
            self,
            config: SupervisedConfig,
            checkpoint: Optional[str | PathLike] = None,
            checkpoint_keys: Optional[Sequence[str]] = None,
            logging: str = "online",
            wandb_id: Optional[str] = None,
            tags: Optional[Sequence[str]] = None,
            group: Optional[str] = None,
            print_model_summary: bool = True,
    ):
        super().__init__(config=config, logging=logging, wandb_id=wandb_id, tags=tags, group=group)

        # Create datasets and data loaders
        self.datasets, self.loaders = self._create_data_loaders(
            batch_size=config.batch_size,
            workers=config.loader_workers,
            datasets=config.dataset
        )

        # Create model and optimizer
        self.model = instantiate(config.model)
        self.optimizer = instantiate(config.pipeline.optimizer, params=self.model.parameters())
        if print_model_summary:
            _print_model_summary(self.model, self.logging_dir)

        # Create learning rate scheduler
        max_iterations = len(self.loaders["train"]) * config.max_epochs
        self.lr_scheduler = create_lr_scheduler(
            optimizer=self.optimizer,
            config=self.pipeline_config.scheduler,
            max_iterations=max_iterations,
        )

        # Wrap with accelerator
        self.model, self.optimizer, self.lr_scheduler = (
            self.accelerator.prepare(self.model, self.optimizer, self.lr_scheduler)
        )
        for split in self.loaders.keys():
            self.loaders[split] = self.accelerator.prepare(self.loaders[split])

        # Create loss function
        self.loss_fn = self._create_composite_loss(config=config.losses)

        # Create trainer
        self.trainer = SupervisedTrainer(
            accelerator=self.accelerator,
            model=self.model,
            config=config,
            loss_fn=self.loss_fn,
            optimizer=self.optimizer,
            scheduler=self.lr_scheduler,
            clip_grad_norm=self.pipeline_config.gradient_accumulation.clip_grad_norm,
            clip_grad_value=self.pipeline_config.gradient_accumulation.clip_grad_value,
        )

        # Attach logger handlers for training
        OutputLogger("train/loss", self.log).attach(self.trainer)
        CompositeLossLogger(self.loss_fn, self.log, "loss/").attach(self.trainer)
        OptimizerLogger(self.optimizer, ["lr"], self.log, "optimizer/").attach(self.trainer)

        # Create evaluator
        self.evaluator = SupervisedEvaluator(
            accelerator=self.accelerator,
            model=self.model,
            config=config
        )
        if "val" in self.loaders:
            self.trainer.add_event_handler(
                event=self.log_interval,
                function=lambda: self.evaluator.run(self.loaders["val"]),
            )
        else:
            warnings.warn('No "val" dataset provided. Validation will not be performed.')

        # Set up metric logging
        self.metrics = {}
        for metric_label, metric_conf in config.validation.metrics.items():
            m = instantiate(metric_conf, sync_on_compute=False)  # Avoid double computation with accelerator
            self.metrics[metric_label] = m.to(self.device)

        # Attach logger handlers for validation
        MetricLogger(metrics=self.metrics, log_function=self.log, prefix="val/").attach(self.evaluator)

        # Set up checkpoint handlers
        to_save = {
            "trainer": self.trainer,
            "model": self.model,
            "optimizer": self.optimizer,
            "scheduler": self.lr_scheduler,
        }
        to_unwrap = ["model"]
        output_folder = f"checkpoints/{self.run_name}"

        for ckpt_def in config.checkpoints:
            score_function = None
            if ckpt_def.metric is not None:
                score_function = partial(lambda metric: metric.compute().item(), metric=self.metrics[ckpt_def.metric])

            checkpoint_handler = Checkpoint(
                accelerator=self.accelerator,
                config=self.config,
                to_save=to_save,
                output_folder=output_folder,
                global_step_function=lambda: self.trainer.step,
                score_function=score_function,
                score_name=ckpt_def.metric,
                score_mode=ckpt_def.mode,
                to_unwrap=to_unwrap,
                max_saved=ckpt_def.num_saved,
            )
            self.trainer.add_event_handler(event=parse_log_interval(ckpt_def.interval), function=checkpoint_handler)

        # Load checkpoint
        if checkpoint is not None:
            self._load_checkpoint(
                path=checkpoint,
                to_load={
                    "trainer": self.trainer,
                    "model": self.model,
                    "optimizer": self.optimizer,
                    "scheduler": self.lr_scheduler,
                },
                to_unwrap=["model"],
                keys=checkpoint_keys
            )

    def run(self):
        """Run pipeline."""
        try:
            self.trainer.run(loader=self.loaders["train"], max_epochs=self.config.max_epochs)
        except KeyboardInterrupt:
            self.print("Interrupted")
        finally:
            self.accelerator.end_training()


def _print_model_summary(model: torch.nn.Module, logging_directory: str = "") -> None:
    """
    Prints the model summary.

    Args:
        model (torch.nn.Module): Model to print the summary for.
        logging_directory (str): Directory to save the model state
    """
    param_kwargs = {}
    trainable_parameters = sum(
        p.numel() for p in model.parameters(**param_kwargs) if p.requires_grad
    )
    total_parameters = sum(p.numel() for p in model.parameters(**param_kwargs))
    non_trainable_parameters = total_parameters - trainable_parameters
    logging.info("==" * 10)
    logging.info(f"Summary for model {type(model)}")
    logging.info(f"Model is {model}")
    logging.info(f"\tTotal parameters {_get_human_readable_count(total_parameters)}")
    logging.info(
        f"\tTrainable parameters {_get_human_readable_count(trainable_parameters)}"
    )
    logging.info(
        f"\tNon-Trainable parameters {_get_human_readable_count(non_trainable_parameters)}"
    )
    logging.info("==" * 10)

    if logging_directory:
        output_path = os.path.join(logging_directory, "model.txt")

        logging.info(f"Saving model summary to {output_path}")

        try:
            with open(output_path, "w") as f:
                print(model, file=f)
        except Exception as e:
            logging.error(f"Error saving model summary: {e}")

        logging.info("Model summary printed.")


def _get_human_readable_count(number: float) -> str:
    """
    Abbreviates an integer number with K, M, B, T for thousands, millions,
    billions and trillions, respectively.
    Examples:
        >>> _get_human_readable_count(123)
        '123  '
        >>> _get_human_readable_count(1234)  # (one thousand)
        '1.2 K'
        >>> _get_human_readable_count(2e6)   # (two million)
        '2.0 M'
        >>> _get_human_readable_count(3e9)   # (three billion)
        '3.0 B'
        >>> _get_human_readable_count(4e14)  # (four hundred trillion)
        '400 T'
        >>> _get_human_readable_count(5e15)  # (more than trillion)
        '5,000 T'
    Args:
        number (float): a positive integer number

    Return:
        str: A string formatted according to the pattern described above.
    """
    assert number >= 0
    labels = [" ", "K", "M", "B", "T"]
    num_digits = int(np.floor(np.log10(number)) + 1 if number > 0 else 1)
    num_groups = int(np.ceil(num_digits / 3))
    num_groups = min(num_groups, len(labels))  # don't abbreviate beyond trillions
    shift = -3 * (num_groups - 1)
    number = number * (10 ** shift)
    index = num_groups - 1

    if index < 1 or number >= 100:
        return f"{int(number):,d} {labels[index]}"
    else:
        return f"{number:,.1f} {labels[index]}"