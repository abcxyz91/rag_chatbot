"""Route CrewAI streaming events to the active chat request."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from queue import Queue
from typing import Iterator, TypeAlias

from crewai.events import BaseEventListener, LLMStreamChunkEvent


@dataclass(frozen=True, slots=True)
class StreamChunk:
    text: str


@dataclass(frozen=True, slots=True)
class StreamComplete:
    response: str


@dataclass(frozen=True, slots=True)
class StreamFailed:
    message: str


StreamMessage: TypeAlias = StreamChunk | StreamComplete | StreamFailed
StreamQueue: TypeAlias = Queue[StreamMessage]

_active_queue: ContextVar[StreamQueue | None] = ContextVar(
    "rag_chatbot_active_stream_queue", default=None
)


class _StreamEventRouter(BaseEventListener):
    """Register one process-wide handler and isolate streams with context state."""

    def setup_listeners(self, crewai_event_bus) -> None:
        @crewai_event_bus.on(LLMStreamChunkEvent)
        def on_llm_stream_chunk(source, event: LLMStreamChunkEvent) -> None:
            del source
            queue = _active_queue.get()
            if queue is not None and event.chunk:
                queue.put(StreamChunk(str(event.chunk)))


# BaseEventListener registers handlers during construction. Keeping one router avoids
# adding another permanent global event-bus handler on every Streamlit rerun.
_event_router = _StreamEventRouter()


@contextmanager
def capture_stream(queue: StreamQueue) -> Iterator[None]:
    """Send CrewAI chunks emitted in this async context to ``queue`` only."""

    token = _active_queue.set(queue)
    try:
        yield
    finally:
        _active_queue.reset(token)
