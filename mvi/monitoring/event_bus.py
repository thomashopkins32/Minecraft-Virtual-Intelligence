from typing import Callable

from .schema.event import Event


class EventBus:
    def __init__(self) -> None:
        self._listeners: dict[Event, list[Callable]] = {}

    def publish(self, event: Event) -> None:
        for listener in self._listeners.get(event, []):
            listener(event)
        
    def subscribe(self, event: Event, callback: Callable) -> None:
        self._listeners.setdefault(event, []).append(callback)