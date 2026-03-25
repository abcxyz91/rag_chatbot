# streamlit_app.py
import streamlit as st
import asyncio
from queue import Queue
from threading import Thread
from streaming_listener import StreamToQueue
from main import RagChatbotFlow

st.title("RAG Chatbot")

if "history" not in st.session_state:
    st.session_state.history = []

for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_input := st.chat_input("Ask a question..."):

    with st.chat_message("user"):
        st.markdown(user_input)

    token_queue = Queue()
    listener = StreamToQueue(token_queue)  # listener is now active

    def run_flow():
        flow = RagChatbotFlow()
        flow.state.user_query = user_input
        asyncio.run(flow.kickoff_async())
        token_queue.put(None)  # ← sentinel: signal stream is done

    thread = Thread(target=run_flow)
    thread.start()

    with st.chat_message("assistant"):
        def token_generator():
            while True:
                chunk = token_queue.get()
                if chunk is None:
                    break
                yield chunk

        full_response = st.write_stream(token_generator())

    thread.join()

    st.session_state.history.extend([
        {"role": "user", "content": user_input},
        {"role": "assistant", "content": full_response}
    ])