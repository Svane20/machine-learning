import sys
from pathlib import Path
from typing import Union
import hydra
from hydra import initialize, compose
from omegaconf import OmegaConf, DictConfig
from omegaconf.errors import ConfigKeyError, MissingMandatoryValue

from ..schemas.config import ConfigType, SupervisedConfig, GANConfig

OmegaConf.register_new_resolver("get_method", hydra.utils.get_method)
OmegaConf.register_new_resolver("get_class", hydra.utils.get_class)
OmegaConf.register_new_resolver("add", lambda x, y: x + y)
OmegaConf.register_new_resolver("times", lambda *args: __import__("numpy").prod(args).item())
OmegaConf.register_new_resolver("divide", lambda x, y: x / y)
OmegaConf.register_new_resolver("pow", lambda x, y: x ** y)
OmegaConf.register_new_resolver("subtract", lambda x, y: x - y)
OmegaConf.register_new_resolver("range", lambda x: list(range(x)))
OmegaConf.register_new_resolver("int", lambda x: int(x))
OmegaConf.register_new_resolver("ceil_int", lambda x: int(__import__("math").ceil(x)))
OmegaConf.register_new_resolver("merge", lambda *x: OmegaConf.merge(*x))

Configuration = Union[SupervisedConfig, GANConfig]


def load_configuration(config_path: Path, version_base: str = "1.2", caller_stack_depth: int = 2) -> Configuration:
    """
    Load the configuration from a YAML file.

    Args:
        config_path (Path): Path to the configuration file.
        version_base (str): Version base for Hydra configuration.
        caller_stack_depth (int): Depth of the stack to find the caller's path.

    Returns:
        Configuration: The loaded and resolved configuration object.
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    config_dir = config_path.parent.as_posix()
    config_name = config_path.name

    with initialize(config_path=config_dir, version_base=version_base, caller_stack_depth=caller_stack_depth):
        cfg = compose(config_name=config_name)

    # Convert to the correct configuration based on the type
    return _load_configuration_based_on_type(cfg)


def _load_configuration_based_on_type(cfg: DictConfig) -> Configuration:
    """
    Load the configuration based on the specified pipeline type.

    Args:
        cfg (DictConfig): The configuration to validate and merge.

    Returns:
        Configuration: The validated and merged configuration object.
    """

    config_type = cfg.type
    if config_type is None:
        raise ValueError("Pipeline type is not specified in the configuration.")

    if config_type == ConfigType.SUPERVISED:
        return _validate_configuration(base=OmegaConf.structured(SupervisedConfig), config=cfg)
    elif config_type == ConfigType.GAN:
        return _validate_configuration(base=OmegaConf.structured(GANConfig), config=cfg)
    else:
        raise ValueError(f"Unsupported pipeline type: {config_type}")


def _validate_configuration(base: Configuration, config: DictConfig) -> Configuration:
    """
    Validate and merge the configuration with the base configuration.

    Args:
        base (Configuration): The base configuration structure to validate against.
        config (DictConfig): The configuration to validate and merge.

    Returns:
        Configuration: The validated and merged configuration object.
    """
    try:
        merged = OmegaConf.merge(base, config)
    except ConfigKeyError as e:
        print(f"Configuration contains invalid keys or types: {e}", file=sys.stderr)
        sys.exit(1)

    OmegaConf.to_container(merged, resolve=True)

    try:
        return OmegaConf.to_object(merged)
    except MissingMandatoryValue as e:
        print(f"Missing mandatory value in configuration: {e}", file=sys.stderr)
        sys.exit(1)
