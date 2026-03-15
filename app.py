import types
import matplotlib.figure
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import numpy as np
from os.path import exists

from data_handler import load_data
from query_engine import QueryEngine
from vector_store import VectorDB


st.set_page_config(layout="wide")
st.title("Fetii Data Chatbot 🚖")
st.write("Ask a question about the Fetii trip data!")

# Load the data directly from the file path
data_path = 'data/Fetii_data.csv'

# Check if the data file exists
if not exists(data_path):
    st.error(f"Error: The data file '{data_path}' was not found. Please ensure it exists in the correct directory.")
else:
    df = load_data(data_path)
    st.success("✅ Data loaded successfully!")

    engine = QueryEngine(df)
    vectordb = VectorDB()

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "cache" not in st.session_state:
        st.session_state.cache = {}

    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            # Use the helper function for display
            if isinstance(message["content"], (pd.DataFrame, matplotlib.figure.Figure, types.ModuleType, str, int, float, np.floating, np.integer)):
                if isinstance(message["content"], types.ModuleType):
                    st.pyplot(message["content"])
                elif isinstance(message["content"], (pd.DataFrame, str, int, float, np.floating, np.integer)):
                    st.markdown(message["content"])
                else:
                    st.pyplot(message["content"])
            else:
                st.markdown(f"Unrecognized content type: `{type(message['content'])}`\n\n{message['content']}")


    if prompt := st.chat_input("Ask a question"):
        # Add user message to chat history and display
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            response_placeholder = st.empty()

            # 🔁 Cache lookup
            if prompt in st.session_state.cache:
                answer = st.session_state.cache[prompt]
            else:
                # 🔎 Vector DB semantic match
                answer = vectordb.search(prompt)
                if answer is None:
                    answer = engine.answer(prompt)
                    vectordb.add(prompt, answer)
                st.session_state.cache[prompt] = answer

            # 🖼️ Display the new answer and save it to history
            if isinstance(answer, pd.DataFrame):
                response_placeholder.dataframe(answer, use_container_width=True)
            elif isinstance(answer, matplotlib.figure.Figure):
                response_placeholder.pyplot(answer)
            elif isinstance(answer, types.ModuleType) and answer.__name__ == "matplotlib.pyplot":
                response_placeholder.pyplot(plt.gcf())
            elif isinstance(answer, (str, int, float, np.floating, np.integer)):
                response_placeholder.markdown(f"**Answer:** {answer}")
            else:
                response_placeholder.markdown(f"Unrecognized result type: `{type(answer)}`\n\n{answer}")

            st.session_state.messages.append({"role": "assistant", "content": answer})
