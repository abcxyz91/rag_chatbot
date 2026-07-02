"""Streamlit chat interface for the RAG chatbot."""

from __future__ import annotations

import asyncio
from queue import Empty
from threading import Thread
from typing import Callable

import streamlit as st

from rag_chatbot.main import RagChatbotFlow
from rag_chatbot.settings import get_settings
from rag_chatbot.streaming_listener import (
    StreamChunk,
    StreamComplete,
    StreamFailed,
    StreamQueue,
    capture_stream,
)

MAX_QUERY_LENGTH = 1_000
MAX_HISTORY_MESSAGES = 20
FAILED_RESPONSE = "Sorry, I couldn't generate a response. Please try again."


def _run_flow(
    user_input: str,
    history: list[dict[str, str]],
    output: StreamQueue,
    flow_factory: Callable[[], RagChatbotFlow] = RagChatbotFlow,
) -> None:
    """Run CrewAI off the Streamlit thread and always report a terminal message."""

    try:
        flow = flow_factory()
        flow.state.user_query = user_input
        flow.state.conversation_history = [dict(message) for message in history][
            -MAX_HISTORY_MESSAGES:
        ]
        with capture_stream(output):
            asyncio.run(flow.kickoff_async())
        output.put(StreamComplete(response=str(flow.state.response or "")))
    except Exception as exc:  # The UI must not hang when a backend fails.
        output.put(StreamFailed(message=str(exc)))


def _stream_response(worker: Thread, output: StreamQueue) -> tuple[str, str | None]:
    """Yield queued chunks to Streamlit and return the final response/error."""

    terminal: dict[str, str | None] = {"response": None, "error": None}

    def chunks():
        while worker.is_alive() or not output.empty():
            try:
                message = output.get(timeout=0.1)
            except Empty:
                continue

            if isinstance(message, StreamChunk):
                yield message.text
            elif isinstance(message, StreamComplete):
                terminal["response"] = message.response
                break
            else:
                terminal["error"] = message.message
                break

    streamed = st.write_stream(chunks())
    worker.join()
    streamed_text = (
        streamed if isinstance(streamed, str) else "".join(map(str, streamed))
    )
    final_response = terminal["response"] or streamed_text
    if final_response and not streamed_text:
        st.markdown(final_response)
    return final_response, terminal["error"]


def _render_sidebar() -> None:
    settings = get_settings()
    with st.sidebar:
        st.subheader("Session")
        st.caption(f"Answer model: `{settings.answer_model}`")
        st.caption("Streaming: on" if settings.streaming_enabled else "Streaming: off")
        if st.button("Clear conversation", use_container_width=True):
            st.session_state.history = []
            st.rerun()


def main() -> None:
    st.set_page_config(page_title="RAG Chatbot", page_icon="💬", layout="centered")
    st.title("RAG Chatbot")
    st.caption("Ask about accounting, HR, legal topics, or general knowledge.")

    try:
        get_settings().validate_startup()
    except (RuntimeError, ValueError) as exc:
        st.error("The chatbot is not configured yet.")
        st.code(str(exc))
        st.info("Update your .env settings, then restart Streamlit.")
        st.stop()

    if "history" not in st.session_state:
        st.session_state.history = []

    _render_sidebar()

    for message in st.session_state.history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_input = st.chat_input("Ask a question…")
    if not user_input:
        return
    if len(user_input) > MAX_QUERY_LENGTH:
        st.warning(f"Please keep your question under {MAX_QUERY_LENGTH:,} characters.")
        return

    prior_history = list(st.session_state.history)
    st.session_state.history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    output = StreamQueue()
    worker = Thread(
        target=_run_flow,
        args=(user_input, prior_history, output),
        daemon=True,
        name="rag-chatbot-flow",
    )
    worker.start()

    with st.chat_message("assistant"):
        response, error = _stream_response(worker, output)
        if error:
            response = FAILED_RESPONSE
            st.error(response)
            with st.expander("Technical details"):
                st.code(error)
        elif not response:
            response = FAILED_RESPONSE
            st.warning(response)

    st.session_state.history.append({"role": "assistant", "content": response})
    st.session_state.history = st.session_state.history[-MAX_HISTORY_MESSAGES:]


if __name__ == "__main__":
    main()
