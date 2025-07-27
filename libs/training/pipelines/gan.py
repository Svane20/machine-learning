import warnings
from functools import partial
from os import PathLike
from typing import Optional, Sequence

from hydra.utils import instantiate

from libs.schemas.config import GANConfig
from libs.training.handlers.checkpoint import Checkpoint
from libs.training.handlers.composite_loss_logger import CompositeLossLogger
from libs.training.handlers.metric_logger import MetricLogger
from libs.training.handlers.optimizer_logger import OptimizerLogger
from libs.training.handlers.output_logger import OutputLogger
from libs.training.helpers.parse_log_interval import parse_log_interval
from libs.training.lr_scheduler import create_lr_scheduler
from libs.training.pipelines.pipeline import Pipeline
from libs.training.trainers.gan import GanTrainer
from libs.training.trainers.supervised import SupervisedEvaluator


class GANPipeline(Pipeline):
    def __init__(
            self,
            config: GANConfig,
            checkpoint: Optional[str | PathLike] = None,
            checkpoint_keys: Optional[Sequence[str]] = None,
            logging: str = "online",
            wandb_id: Optional[str] = None,
            tags: Optional[Sequence[str]] = None,
            group: Optional[str] = None,
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

        # Create discriminator model
        self.disc_model = instantiate(config.disc_model)
        self.disc_optimizer = instantiate(config=config.pipeline.disc_optimizer, params=self.disc_model.parameters())

        # Create learning rate schedulers
        max_iterations = len(self.loaders["train"]) * config.max_epochs
        self.lr_scheduler = create_lr_scheduler(
            optimizer=self.optimizer,
            config=self.pipeline_config.scheduler,
            max_iterations=max_iterations,
        )
        self.disc_lr_scheduler = create_lr_scheduler(
            optimizer=self.disc_optimizer,
            config=self.pipeline_config.disc_scheduler,
            max_iterations=max_iterations,
        )

        # Wrap with accelerator
        self.model, self.optimizer, self.lr_scheduler = (
            self.accelerator.prepare(
                self.model, self.optimizer, self.lr_scheduler
            )
        )
        self.disc_model, self.disc_optimizer, self.disc_lr_scheduler = (
            self.accelerator.prepare(
                self.disc_model, self.disc_optimizer, self.disc_lr_scheduler
            )
        )
        for split in self.loaders.keys():
            self.loaders[split] = self.accelerator.prepare(self.loaders[split])

        # Create loss functions
        self.loss_fn = self._create_composite_loss(config.losses)
        self.disc_loss_fn = self._create_composite_loss(config.disc_losses)

        # Create trainer
        self.trainer = GanTrainer(
            accelerator=self.accelerator,
            config=config,
            model=self.model,
            disc_model=self.disc_model,
            loss_fn=self.loss_fn,
            disc_loss_fn=self.disc_loss_fn,
            optimizer=self.optimizer,
            disc_optimizer=self.disc_optimizer,
            scheduler=self.lr_scheduler,
            disc_scheduler=self.disc_lr_scheduler,
            clip_grad_norm=self.pipeline_config.gradient_accumulation.clip_grad_norm,
            clip_grad_value=self.pipeline_config.gradient_accumulation.clip_grad_value,
        )

        # Attach logger handlers for training
        OutputLogger("train/loss", self.log, lambda o: o[0]).attach(self.trainer)
        OutputLogger("train/disc_loss", self.log, lambda o: o[1]).attach(self.trainer)
        CompositeLossLogger(self.loss_fn, self.log, "loss/").attach(self.trainer)
        CompositeLossLogger(self.disc_loss_fn, self.log, "disc_loss/").attach(self.trainer)
        OptimizerLogger(self.optimizer, ["lr"], self.log, "optimizer/").attach(self.trainer)
        OptimizerLogger(self.disc_optimizer, ["lr"], self.log, "disc_optimizer/").attach(self.trainer)

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
            "disc_model": self.disc_model,
            "optimizer": self.optimizer,
            "disc_optimizer": self.disc_optimizer,
            "scheduler": self.lr_scheduler,
            "disc_scheduler": self.disc_lr_scheduler,
        }
        to_unwrap = ["model", "disc_model"]
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
                to_load=to_save,
                to_unwrap=to_unwrap,
                keys=checkpoint_keys,
            )

    def run(self) -> None:
        """Run pipeline."""
        try:
            self.trainer.run(loader=self.loaders["train"], max_epochs=self.config.max_epochs)
        except KeyboardInterrupt:
            self.print("Interrupted")
        finally:
            self.accelerator.end_training()
