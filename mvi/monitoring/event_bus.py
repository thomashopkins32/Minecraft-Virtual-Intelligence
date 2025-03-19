from typing import Callable, Dict, List, Type, TypeVar, Set

from .event import Event

T = TypeVar("T", bound=Event)


class EventBus:
    def __init__(self) -> None:
        self._listeners: Dict[Type[Event], List[Callable]] = {}
        self._disabled_listeners: Set[Callable] = set()
        self._enabled = True  # Global toggle

    def publish(self, event: Event) -> None:
        if not self._enabled:
            return

        event_type = type(event)
        for listener in self._listeners.get(event_type, []):
            if listener not in self._disabled_listeners:
                listener(event)

    def subscribe(
        self, event_type: Type[T]
    ) -> Callable[[Callable[[T], None]], Callable[[T], None]]:
        def decorator(callback: Callable[[T], None]) -> Callable[[T], None]:
            self._listeners.setdefault(event_type, []).append(callback)
            return callback

        return decorator

    def disable(self) -> None:
        """Disable all event publishing"""
        self._enabled = False

    def enable(self) -> None:
        """Enable event publishing"""
        self._enabled = True

    def disable_listener(self, listener: Callable) -> None:
        """Disable a specific listener"""
        self._disabled_listeners.add(listener)

    def enable_listener(self, listener: Callable) -> None:
        """Enable a previously disabled listener"""
        if listener in self._disabled_listeners:
            self._disabled_listeners.remove(listener)


event_bus = EventBus()
