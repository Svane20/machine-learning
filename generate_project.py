import argparse
import shutil
from pathlib import Path
import logging
import sys
from typing import Optional, Sequence
import re

from accelerate.utils import DynamoBackend
from omegaconf import OmegaConf

from libs.schemas.config import ConfigType, LogInterval, AMPConfig, GradientAccumulationConfig, SchedulerConfig, \
    SupervisedConfig, CheckpointConfig, DynamoConfig, CheckpointMode, ValidationConfig, LoggingConfig, IntervalStrategy, \
    GANConfig
from libs.schemas.events import EventStep

# DIRECTORIES
CURRENT_DIR = Path(__file__).parent
PROJECTS_DIR = CURRENT_DIR / 'projects'
TEMPLATES_DIR = CURRENT_DIR / 'libs' / 'templates'


def _parse_arguments(args: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """
    Parse command line arguments for the project generator.

    Args:
        args (Optional[Sequence[str]]): List of command line arguments.

    Returns:
        argparse.Namespace: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Generate a new project"
    )
    parser.add_argument(
        "-n", "--name",
        metavar="NAME",
        help="Name of the project to create.",
        default="test",
        type=str
    )
    parser.add_argument(
        '-d', '--dir',
        metavar='DIR',
        help="Parent directory where the project folder will be created.",
        default=PROJECTS_DIR,
        type=Path
    )
    parser.add_argument(
        "-t", "--type",
        metavar="TYPE",
        help="Pipeline type to create",
        choices=["supervised", "gan"],
        default="gan",
        type=str
    )
    parser.add_argument(
        "-v", "--verbose",
        action="count",
        default=0,
        help="Increase verbosity (use -v or -vv)."
    )
    return parser.parse_args(args)


def _add_header(path: Path, header: str = "# @package _global_") -> None:
    """
    Prepend a comment header and a blank line to the top of the file.
    """
    content = path.read_text()
    # header + one blank line + existing content
    path.write_text(f"{header}\n\n{content}")


def _wrap_only_strings(path: Path):
    """Wrap only unquoted YAML scalar strings in double-quotes."""
    text = path.read_text().splitlines()
    out = []
    line_re = re.compile(r"^(\s*[^:\s][^:]*:\s*)(\S.*)$")
    num_re = re.compile(r"^[+-]?\d+(\.\d*)?([eE][+-]?\d+)?$")
    specials = {"null", "true", "false"}  # YAML booleans/null

    for ln in text:
        m = line_re.match(ln)
        if m:
            key, val = m.groups()
            if (not (val.startswith(('"', "'")) or val.lower() in specials
                     or num_re.match(val) or val.startswith("- "))):
                ln = f'{key}"{val}"'
        out.append(ln)

    path.write_text("\n".join(out) + "\n")


def _prettify_yaml(path: Path):
    """
    - Inserts blank lines before nested blocks (except our special metrics→checkpoints case).
    - Adds one blank after the first pipeline child.
    - Guarantees exactly one blank between metrics: and checkpoints:.
    """
    lines = path.read_text().splitlines()
    out = []
    in_pipeline = False
    first_child = 0
    last_was_metrics = False

    def indent(line):
        return len(line) - len(line.lstrip(" "))

    for i, line in enumerate(lines):
        il = indent(line)
        stripped = line.lstrip()
        # Lookahead info
        nxt = lines[i + 1] if i + 1 < len(lines) else ""
        nxt_stripped = nxt.lstrip()
        nxt_indent = indent(nxt)

        # 1) Top-level blank-before-block, but skip for our metrics→checkpoints edge
        if il == 0 and out and nxt_indent > 0 and not nxt_stripped.startswith("-"):
            if not (last_was_metrics and nxt_stripped.startswith("checkpoints:")):
                out.append("")

        out.append(line)

        # 2) Detect and handle pipeline first-child blank
        if il == 0 and stripped.startswith("pipeline:"):
            in_pipeline, first_child = True, 0
        elif in_pipeline:
            if il == 2:
                first_child += 1
                if first_child == 1:
                    out.append("")  # after the very first child
            elif il == 0 and stripped:
                in_pipeline = False

        # 3) Handle exactly one blank between metrics: and checkpoints:
        if stripped == "metrics:":
            last_was_metrics = True
        elif last_was_metrics:
            # We just passed the metrics: block; check if this line is checkpoints:
            if stripped.startswith("checkpoints:"):
                # Ensure exactly one blank in out right before this checkpoints:
                # - if the last line in out is already blank, do nothing
                if out and out[-1].strip() != "":
                    out.insert(-1, "")
            # Clear the flag no matter what (we only want it once)
            last_was_metrics = False

    # Write back (preserve final newline)
    path.write_text("\n".join(out) + "\n")


def _ensure_metrics_to_checkpoints_spacing(path: Path):
    """
    Ensures exactly one blank line between the end of the metrics: block
    and the checkpoints: key, without touching anything else.
    """
    text = path.read_text()

    # This pattern finds "metrics:" up through the last non-blank line
    # before "checkpoints:", then collapses whatever whitespace is there
    # into exactly two newlines.
    fixed = re.sub(
        r"(metrics:.*?)(\n\s*\n|\n+)(checkpoints:)",
        lambda m: f"{m.group(1).rstrip()}\n\n{m.group(3)}",
        text,
        flags=re.DOTALL
    )

    path.write_text(fixed)


def _remove_blank_after_checkpoints(path: Path):
    """
    Collapse any number of blank lines right after 'checkpoints:'
    into exactly one newline, so the list item comes immediately after.
    """
    text = path.read_text()
    # Match 'checkpoints:' then any whitespace/newlines, then the first '-'
    fixed = re.sub(
        r'(checkpoints:\s*\n)(\s*\n+)+(\s*-\s)',
        lambda m: f"{m.group(1)}{m.group(3)}",
        text
    )
    path.write_text(fixed)


def _configure_logging(verbosity: int):
    level = logging.WARNING if verbosity == 0 else (logging.INFO if verbosity == 1 else logging.DEBUG)
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(message)s")


def _copy_tree(src: Path, dst: Path):
    for item in src.rglob("*"):
        rel = item.relative_to(src)
        target = dst / rel
        if item.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        else:
            if not target.exists():
                shutil.copy2(item, target)
                logging.info("Copied %s → %s", item, target)


def _build_supervised_cfg(name: str) -> SupervisedConfig:
    """Populate a Structured SupervisedConfig with sensible defaults."""
    cfg = OmegaConf.structured(SupervisedConfig)

    # Base fields
    cfg.type = ConfigType.SUPERVISED.value
    cfg.experiment_name = name

    # Datasets
    for split in ("train", "val"):
        cfg.dataset[split] = {"_target_": f"datasets.example.ExampleDataset"}

    # Model
    cfg.model = {"_target_": "models.example.ExampleModel"}

    # Pipeline
    cfg.pipeline.seed = None
    cfg.pipeline.amp = AMPConfig(enabled=True, type="fp16")
    cfg.pipeline.dynamo = DynamoConfig(enabled=False, backend=DynamoBackend.INDUCTOR.value.lower())
    cfg.pipeline.gradient_accumulation = GradientAccumulationConfig()
    cfg.pipeline.scheduler = SchedulerConfig(type="cosine", end_value=1e-7, warmup_steps=2)
    cfg.pipeline.optimizer = {"_target_": "torch.optim.AdamW", "lr": 3e-4}

    # Losses
    cfg.losses = {"L1": {"_target_": "torch.nn.L1Loss", "weight": 1.0}}

    # Validation
    cfg.validation = ValidationConfig(
        event=EventStep.EPOCH_COMPLETED.value,
        metrics={"SSIM": {"_target_": "torchmetrics.image.StructuralSimilarityIndexMeasure", "data_range": 1.0}}
    )

    # Checkpoints
    cfg.checkpoints = [
        CheckpointConfig(
            metric="SSIM",
            mode=CheckpointMode.MAX.value,
            num_saved=3,
            interval=LogInterval(event=EventStep.EPOCH_COMPLETED.value, every=1)
        )
    ]

    # Logging
    cfg.logging = LoggingConfig(
        dir="logs",
        strategy=IntervalStrategy.STEPS.value,
        every=500
    )

    return cfg


def _build_gan_cfg(name: str) -> GANConfig:
    """Populate a Structured GANConfig with sensible defaults."""
    cfg = OmegaConf.structured(GANConfig)

    # Base fields
    cfg.type = ConfigType.GAN.value
    cfg.experiment_name = name

    # Datasets
    for split in ("train", "val"):
        cfg.dataset[split] = {"_target_": f"datasets.example.ExampleDataset"}

    # Models
    cfg.model = {"_target_": "models.example.ExampleModel"}
    cfg.disc_model = {"_target_": "models.disc.ExampleDiscModel"}

    # Pipeline
    cfg.pipeline.seed = None
    cfg.pipeline.amp = AMPConfig(enabled=True, type="fp16")
    cfg.pipeline.dynamo = DynamoConfig(enabled=False, backend=DynamoBackend.INDUCTOR.value.lower())
    cfg.pipeline.gradient_accumulation = GradientAccumulationConfig()
    cfg.pipeline.scheduler = SchedulerConfig(type="cosine", end_value=1e-7, warmup_steps=0)
    cfg.pipeline.disc_scheduler = SchedulerConfig(type="cosine", end_value=1e-7, warmup_steps=0)
    cfg.pipeline.optimizer = {"_target_": "torch.optim.AdamW", "lr": 3e-4, "betas": [0.5, 0.999]}
    cfg.pipeline.disc_optimizer = {"_target_": "torch.optim.AdamW", "lr": 3e-4, "betas": [0.5, 0.999]}

    # Losses
    cfg.losses = {"L1": {"_target_": "torch.nn.L1Loss", "weight": 1.0}}
    cfg.disc_losses = {"L1": {"_target_": "torch.nn.L1Loss", "weight": 1.0}}

    # Validation
    cfg.validation = ValidationConfig(
        event=EventStep.EPOCH_COMPLETED.value,
        metrics={"PSNR": {"_target_": "torchmetrics.image.PeakSignalNoiseRatio", "data_range": 1.0}}
    )

    # Checkpoints
    cfg.checkpoints = [
        CheckpointConfig(
            metric="PSNR",
            mode=CheckpointMode.MAX.value,
            num_saved=1,
            interval=LogInterval(event=EventStep.EPOCH_COMPLETED.value, every=1)
        )
    ]

    # Logging
    cfg.logging = LoggingConfig(
        dir="logs",
        strategy=IntervalStrategy.STEPS.value,
        every=500
    )

    return cfg


def _create_config_file(dest: Path, name: str, kind: str):
    conf_dir = (dest / "configs")
    conf_dir.mkdir(parents=True, exist_ok=True)
    out_path = conf_dir / "training.yaml"

    if kind == "supervised":
        cfg = _build_supervised_cfg(name)
    elif kind == "gan":
        cfg = _build_gan_cfg(name)
    else:
        raise ValueError(f"Unsupported pipeline type: {kind}")

    OmegaConf.save(config=cfg, f=out_path, resolve=True)
    _prettify_yaml(out_path)
    _wrap_only_strings(out_path)
    _add_header(out_path)
    _ensure_metrics_to_checkpoints_spacing(out_path)
    _remove_blank_after_checkpoints(out_path)
    logging.info("Wrote config → %s", out_path)


def _create_project(parent: Path, name: str, kind: str):
    project_path = parent / name
    project_path.mkdir(parents=True, exist_ok=True)

    # 1) Copy template tree
    _copy_tree(TEMPLATES_DIR / kind, project_path)

    # 2) Generate YAML config
    _create_config_file(project_path, name, kind)

    logging.info("Project scaffolded at %s", project_path)


if __name__ == "__main__":
    args = _parse_arguments()
    _configure_logging(args.verbose)

    try:
        args.dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logging.error("Cannot create parent directory %s: %s", args.dir, e)
        sys.exit(1)

    _create_project(args.dir, args.name, args.type)
