# streaming_listener.py
from crewai.events import LLMStreamChunkEvent, BaseEventListener
from queue import Queue

class StreamToQueue(BaseEventListener):
    def __init__(self, queue: Queue):
        super().__init__()
        self.queue = queue

    def setup_listeners(self, crewai_event_bus):
        @crewai_event_bus.on(LLMStreamChunkEvent)
        def on_llm_stream_chunk(source, event: LLMStreamChunkEvent):
            if event.chunk:  # avoid pushing empty chunks
                self.queue.put(event.chunk)