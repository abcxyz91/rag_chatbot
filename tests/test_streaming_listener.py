import importlib
import sys
from queue import Empty, Queue
from threading import Thread
from types import ModuleType, SimpleNamespace


class FakeEventBus:
    def __init__(self):
        self.handlers = []

    def on(self, event_type):
        del event_type

        def register(handler):
            self.handlers.append(handler)
            return handler

        return register


def import_listener(monkeypatch):
    event_bus = FakeEventBus()

    class BaseEventListener:
        def __init__(self):
            self.setup_listeners(event_bus)

    events = ModuleType("crewai.events")
    events.BaseEventListener = BaseEventListener
    events.LLMStreamChunkEvent = type("LLMStreamChunkEvent", (), {})
    crewai = ModuleType("crewai")
    crewai.events = events
    monkeypatch.setitem(sys.modules, "crewai", crewai)
    monkeypatch.setitem(sys.modules, "crewai.events", events)
    sys.modules.pop("rag_chatbot.streaming_listener", None)

    return importlib.import_module("rag_chatbot.streaming_listener"), event_bus


def test_capture_stream_routes_chunks_only_inside_active_context(monkeypatch):
    listener, event_bus = import_listener(monkeypatch)
    output = Queue()
    handler = event_bus.handlers[0]

    handler(None, SimpleNamespace(chunk="ignored"))
    with listener.capture_stream(output):
        handler(None, SimpleNamespace(chunk="hello"))
        handler(None, SimpleNamespace(chunk=""))
    handler(None, SimpleNamespace(chunk="also ignored"))

    assert output.get_nowait() == listener.StreamChunk("hello")
    try:
        output.get_nowait()
    except Empty:
        pass
    else:
        raise AssertionError("inactive or empty chunks must not be queued")


def test_nested_capture_restores_the_outer_request(monkeypatch):
    listener, event_bus = import_listener(monkeypatch)
    outer = Queue()
    inner = Queue()
    handler = event_bus.handlers[0]

    with listener.capture_stream(outer):
        handler(None, SimpleNamespace(chunk="outer one"))
        with listener.capture_stream(inner):
            handler(None, SimpleNamespace(chunk="inner"))
        handler(None, SimpleNamespace(chunk="outer two"))

    assert [outer.get_nowait().text, outer.get_nowait().text] == [
        "outer one",
        "outer two",
    ]
    assert inner.get_nowait().text == "inner"


def test_concurrent_requests_do_not_mix_chunks(monkeypatch):
    listener, event_bus = import_listener(monkeypatch)
    first = Queue()
    second = Queue()
    handler = event_bus.handlers[0]

    def emit(output, prefix):
        with listener.capture_stream(output):
            for index in range(25):
                handler(None, SimpleNamespace(chunk=f"{prefix}-{index}"))

    workers = [
        Thread(target=emit, args=(first, "first")),
        Thread(target=emit, args=(second, "second")),
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()

    first_chunks = [first.get_nowait().text for _ in range(25)]
    second_chunks = [second.get_nowait().text for _ in range(25)]
    assert all(chunk.startswith("first-") for chunk in first_chunks)
    assert all(chunk.startswith("second-") for chunk in second_chunks)
