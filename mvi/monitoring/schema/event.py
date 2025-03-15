from dataclasses import dataclass
from datetime import datetime

@dataclass
class Event:
    """
    Base class for an event. Contains attributes common to all events.

    Attributes
    ----------
    timestamp: datetime
        The time that the event occurred.
    """
    timestamp: datetime