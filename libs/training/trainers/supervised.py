from typing import Callable, Any, Optional

import torch
from accelerate import Accelerator

from libs.schemas.config import Config
from libs.schemas.train_utils import Phase
from libs.training.trainers.trainer import Trainer


class SupervisedTrainer(Trainer):
    def __init__(
            self,
            accelerator: Accelerator,
            config: Config,
            model: torch.nn.Module,
            loss_fn: Callable[[Any, Any], Any],
            optimizer: torch.optim.Optimizer,
            scheduler: torch.optim.lr_scheduler.LRScheduler,
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
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.clip_grad_norm = clip_grad_norm
        self.clip_grad_value = clip_grad_value

    def process(self, batch):
        self.model.train()

        inputs, targets = batch

        with self.accelerator.accumulate(self.model):
            self.optimizer.zero_grad()

            with self.accelerator.autocast():
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)

            self.accelerator.backward(loss)

            if self.accelerator.sync_gradients:
                if self.clip_grad_norm:
                    self.accelerator.clip_grad_norm_(
                        parameters=self.model.parameters(),
                        max_norm=self.clip_grad_norm
                    )
                if self.clip_grad_value:
                    self.accelerator.clip_grad_value_(
                        parameters=self.model.parameters(),
                        clip_value=self.clip_grad_value
                    )

                self.optimizer.step()
                self.scheduler.step()

        return inputs, targets, outputs, loss


class SupervisedEvaluator(Trainer):
    def __init__(
            self,
            accelerator: Accelerator,
            model: torch.nn.Module,
            config: Config,
    ):
        super().__init__(
            process_fn=self.process,
            logging_conf=config.logging,
            show_progress=config.show_progress,
            phase=Phase.VAL
        )

        self.accelerator = accelerator
        self.model = model

    def process(self, batch):
        self.model.eval()

        inputs, targets = batch

        with torch.no_grad():
            outputs = self.model(inputs)

        outputs, targets = self.accelerator.gather_for_metrics((outputs, targets))

        return outputs, targets
