from typing import Optional, Sequence
from pathlib import Path
import argparse

from libs.training.callbacks.image_logger import ImageLogger
from libs.configuration.utils import load_configuration
from libs.schemas.config import GANConfig
from libs.training.pipelines.gan import GANPipeline


def parse_arguments(arguments: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """
    Parse command line arguments for the training script.

    Args:
        arguments (Optional[Sequence[str]]): Command line arguments to parse.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=Path, default="configs/training.yaml")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--checkpoint-keys", type=str, nargs="+")
    parser.add_argument("--logging", type=str, choices=["online", "offline"], default="online")
    parser.add_argument("--wandb-id", type=str, required=False)
    parser.add_argument("--tags", type=str, nargs="+")
    parser.add_argument("--group", type=str)
    return parser.parse_args(arguments)


if __name__ == "__main__":
    args = parse_arguments()
    config: GANConfig = load_configuration(args.config)

    pipeline = GANPipeline(
        config=config,
        checkpoint=args.checkpoint,
        checkpoint_keys=args.checkpoint_keys,
        logging=args.logging,
        wandb_id=args.wandb_id,
        tags=args.tags,
        group=args.group
    )

    ds_conf = config.dataset["train"]
    image_logger = ImageLogger(
        denormalize_input=True,
        normalize_mean=ds_conf["normalize_mean"],
        normalize_std=ds_conf["normalize_std"],
    )
    pipeline.install_callback(pipeline.log_interval, image_logger)

    pipeline.run()
