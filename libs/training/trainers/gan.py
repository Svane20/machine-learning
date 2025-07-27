from typing import Callable, Any, Optional
import torch
from accelerate import Accelerator

from libs.schemas.config import Config
from libs.schemas.train_utils import Phase
from libs.training.trainers.trainer import Trainer


class GanTrainer(Trainer):
    def __init__(
            self,
            accelerator: Accelerator,
            config: Config,
            model: torch.nn.Module,
            disc_model: torch.nn.Module,
            loss_fn: Callable[..., Any],
            disc_loss_fn: Callable[..., Any],
            optimizer: torch.optim.Optimizer,
            disc_optimizer: torch.optim.Optimizer,
            scheduler: torch.optim.lr_scheduler.LRScheduler,
            disc_scheduler: torch.optim.lr_scheduler.LRScheduler,
            clip_grad_norm: Optional[float] = None,
            clip_grad_value: Optional[float] = None,
    ):
        super().__init__(
            process_fn=self.process,
            logging_conf=config.logging,
            show_progress=config.show_progress,
            phase=Phase.TRAIN
        )

        self.accelerator = accelerator
        self.model = model
        self.disc_model = disc_model
        self.loss_fn = loss_fn
        self.disc_loss_fn = disc_loss_fn
        self.optimizer = optimizer
        self.disc_optimizer = disc_optimizer
        self.scheduler = scheduler
        self.disc_scheduler = disc_scheduler

        self.clip_grad_norm = clip_grad_norm
        self.clip_grad_value = clip_grad_value

    def process(self, batch):
        self.model.train()
        self.disc_model.train()

        inputs, targets = batch

        # Update discriminator
        with self.accelerator.accumulate(self.disc_model):
            self.disc_optimizer.zero_grad()

            with self.accelerator.autocast():
                outputs = self.model(inputs).detach()
                disc_pred_real = self.disc_model(targets)
                disc_pred_fake = self.disc_model(outputs)

                disc_loss = self.disc_loss_fn(outputs, targets, disc_real=disc_pred_real, disc_fake=disc_pred_fake)
            self.accelerator.backward(disc_loss)

            if self.accelerator.sync_gradients:
                if self.clip_grad_norm:
                    self.accelerator.clip_grad_norm_(
                        parameters=self.disc_model.parameters(),
                        max_norm=self.clip_grad_norm,
                    )
                if self.clip_grad_value:
                    self.accelerator.clip_grad_value_(
                        parameters=self.disc_model.parameters(),
                        clip_value=self.clip_grad_value,
                    )

                self.disc_optimizer.step()
                self.disc_scheduler.step()

        # Update generator
        with self.accelerator.accumulate(self.model):
            self.optimizer.zero_grad()

            with self.accelerator.autocast():
                outputs = self.model(inputs)
                disc_pred_fake = self.disc_model(outputs)

                loss = self.loss_fn(outputs, targets, disc_fake=disc_pred_fake)
            self.accelerator.backward(loss)

            if self.accelerator.sync_gradients:
                if self.clip_grad_norm:
                    self.accelerator.clip_grad_norm_(
                        parameters=self.model.parameters(),
                        max_norm=self.clip_grad_norm,
                    )
                if self.clip_grad_value:
                    self.accelerator.clip_grad_value_(
                        parameters=self.model.parameters(),
                        clip_value=self.clip_grad_value,
                    )

                self.optimizer.step()
                self.scheduler.step()

        return inputs, targets, outputs, loss, disc_loss