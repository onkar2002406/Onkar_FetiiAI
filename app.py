# import types
# import matplotlib.figure
# import matplotlib
# import matplotlib.pyplot as plt
# import pandas as pd
# import streamlit as st
# from data_handler import load_data
# from query_engine import QueryEngine
# from vector_store import VectorDB


# st.title("Fetii Data Chatbot 🚖")
# st.write("Upload a CSV file and ask questions about the data!")

# uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

# if uploaded_file:
#     df = load_data(uploaded_file)
#     st.success("✅ Data loaded successfully!")

#     engine = QueryEngine(df)
#     vectordb = VectorDB()

#     if "messages" not in st.session_state:
#         st.session_state.messages = []
#     if "cache" not in st.session_state:
#         st.session_state.cache = {}

#     for m in st.session_state.messages:
#         with st.chat_message(m["role"]):
#             st.markdown(m["content"])

#     if prompt := st.chat_input("Ask a question"):
#         st.session_state.messages.append({"role": "user", "content": prompt})
#         with st.chat_message("user"):
#             st.markdown(prompt)

#         with st.chat_message("assistant"):
#             placeholder = st.empty()

#             # 🔁 Cache lookup
#             if prompt in st.session_state.cache:
#                 ans = st.session_state.cache[prompt]
#             else:
#                 # 🔎 Vector DB semantic match
#                 ans = vectordb.search(prompt)
#                 if ans is None:
#                     ans = engine.answer(prompt)
#                     vectordb.add(prompt, ans)
#                 st.session_state.cache[prompt] = ans

#             # 🖼️ Display answer
#             if isinstance(ans, pd.DataFrame):
#                 placeholder.dataframe(ans)

#             elif isinstance(ans, matplotlib.figure.Figure):
#                 # a real Matplotlib figure
#                 placeholder.pyplot(ans)

#             elif isinstance(ans, types.ModuleType) and ans.__name__ == "matplotlib.pyplot":
#                 # sometimes PandasAI returns the pyplot module itself
#                 placeholder.pyplot(plt.gcf())

#             elif isinstance(ans, str):
#                 # ✅ plain text explanation or error string
#                 placeholder.markdown(ans)

#             else:
#                 # catch anything unexpected
#                 placeholder.markdown(f"Unrecognized result type: {type(ans)}\n\n{ans}")
                
#         st.session_state.messages.append({"role": "assistant", "content": str(ans)})

#     st.sidebar.markdown(f"⚠️ Error Counter: **{engine.error_counter}**")

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
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            # Use the helper function for display
            if isinstance(m["content"], (pd.DataFrame, matplotlib.figure.Figure, types.ModuleType, str, int, float, np.floating, np.integer)):
                if isinstance(m["content"], types.ModuleType):
                    st.pyplot(m["content"])
                elif isinstance(m["content"], (pd.DataFrame, str, int, float, np.floating, np.integer)):
                    st.markdown(m["content"])
                else:
                    st.pyplot(m["content"])
            else:
                st.markdown(f"Unrecognized content type: `{type(m['content'])}`\n\n{m['content']}")


    if prompt := st.chat_input("Ask a question"):
        # Add user message to chat history and display
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            placeholder = st.empty()

            # 🔁 Cache lookup
            if prompt in st.session_state.cache:
                ans = st.session_state.cache[prompt]
            else:
                # 🔎 Vector DB semantic match
                ans = vectordb.search(prompt)
                if ans is None:
                    ans = engine.answer(prompt)
                    vectordb.add(prompt, ans)
                st.session_state.cache[prompt] = ans

            # 🖼️ Display the new answer and save it to history
            if isinstance(ans, pd.DataFrame):
                placeholder.dataframe(ans, use_container_width=True)
            elif isinstance(ans, matplotlib.figure.Figure):
                placeholder.pyplot(ans)
            elif isinstance(ans, types.ModuleType) and ans.__name__ == "matplotlib.pyplot":
                placeholder.pyplot(plt.gcf())
            elif isinstance(ans, (str, int, float, np.floating, np.integer)):
                placeholder.markdown(f"**Answer:** {ans}")
            else:
                placeholder.markdown(f"Unrecognized result type: `{type(ans)}`\n\n{ans}")
                
            st.session_state.messages.append({"role": "assistant", "content": ans})

    st.sidebar.markdown(f"⚠️ Error Counter: **{engine.error_counter}**")