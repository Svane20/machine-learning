from libs.schemas.config import LogInterval
from libs.schemas.events import MatchableEvent, Event


def parse_log_interval(e: LogInterval) -> MatchableEvent:
    """Create matchable event from log interval configuration."""
    if isinstance(e, LogInterval):
        return Event(
            event=e.event,
            every=e.every,
            first=e.first,
            last=e.last,
        )

    raise ValueError(f"Cannot parse log interval {e}.")