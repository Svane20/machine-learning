import os
import logging


class Phase:
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


def makedir(directory_path: str) -> bool:
    """
    Create a directory if it does not exist.

    Args:
        directory_path (str): Directory path.

    Returns:
        bool: Whether the directory was created successfully.
    """
    is_success = False

    try:
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
        is_success = True
    except BaseException as e:
        logging.error(f"Error creating directory: {directory_path}", e)

    return is_success


def human_readable_time(time_seconds: float) -> str:
    """
    Convert seconds to human-readable time.

    Args:
        time_seconds (float): Time in seconds.

    Returns:
        str: Human readable time.
    """
    time = int(time_seconds)
    minutes, seconds = divmod(time, 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)

    return f"{days:02}d {hours:02}h {minutes:02}m"
