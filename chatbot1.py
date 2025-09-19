# import pandas as pd
# import streamlit as st
# from pandasai import SmartDataframe
# from pandasai.llm import OpenAI
# import os

# # --- Streamlit UI and Logic ---
# st.title("Fetii Data Chatbot")
# st.write("Upload a CSV file and ask questions about the data!")

# # File uploader widget
# uploaded_file = st.file_uploader("Upload your data CSV file", type=['csv'])

# if uploaded_file is not None:
#     try:
#         # Read the uploaded CSV into a pandas DataFrame
#         df = pd.read_csv(uploaded_file)
#         st.success("File uploaded and data loaded successfully!")

#         llm = OpenAI()
#         sdf = SmartDataframe(df, config={"llm": llm, "enable_cache": False})

#         # Initialize chat history
#         if "messages" not in st.session_state:
#             st.session_state.messages = []

#         # Display chat messages from history on app rerun
#         for message in st.session_state.messages:
#             with st.chat_message(message["role"]):
#                 st.markdown(message["content"])

#         # Accept user input for the chatbot
#         if prompt := st.chat_input("What would you like to know?"):
#             # Add user message to chat history and display it
#             st.session_state.messages.append({"role": "user", "content": prompt})
#             with st.chat_message("user"):
#                 st.markdown(prompt)

#             # Display assistant response
#             with st.chat_message("assistant"):
#                 message_placeholder = st.empty()
#                 full_response = ""
#                 with st.spinner("Analyzing data..."):
#                     try:
#                         # Use PandasAI to run the query
#                         response = sdf.chat(prompt)

#                         # Check the type of the response and format accordingly
#                         if isinstance(response, pd.DataFrame):
#                             message_placeholder.dataframe(response, use_container_width=True)
#                             full_response = response.to_string() # Change made here
#                         else:
#                             message_placeholder.markdown(str(response))
#                             full_response = str(response)

#                     except Exception as e:
#                         full_response = f"An error occurred: {e}"
#                         message_placeholder.markdown(full_response)

#                 # Add assistant response to chat history
#                 st.session_state.messages.append({"role": "assistant", "content": full_response})

#     except Exception as e:
#         st.error(f"An error occurred while loading the file: {e}")


import os
import pandas as pd
import streamlit as st
from pandasai import SmartDataframe
from pandasai.llm import OpenAI
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# ---------------------- CONFIG ----------------------
st.set_page_config(page_title="Fetii Data Chatbot", layout="wide")
st.title("📊 Fetii Data Chatbot")
st.write("Upload a CSV file and ask questions about the data!")

# ---------------------- STATE -----------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "query_cache" not in st.session_state:
    st.session_state.query_cache = {}

if "vector_index" not in st.session_state:
    st.session_state.vector_index = None
    st.session_state.vector_questions = []
    st.session_state.vector_answers = []

# Sentence-BERT model for vector search
if "embedder" not in st.session_state:
    st.session_state.embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ---------------------- HELPERS ---------------------
@st.cache_data
def load_csv(uploaded_file):
    return pd.read_csv(uploaded_file)

def build_vector_db():
    if st.session_state.vector_questions:
        dim = st.session_state.embedder.get_sentence_embedding_dimension()
        index = faiss.IndexFlatL2(dim)
        embeddings = st.session_state.embedder.encode(
            st.session_state.vector_questions
        )
        index.add(np.array(embeddings).astype("float32"))
        st.session_state.vector_index = index

def search_similar(query, top_k=1, threshold=0.85):
    if not st.session_state.vector_index:
        return None
    q_emb = st.session_state.embedder.encode([query]).astype("float32")
    D, I = st.session_state.vector_index.search(q_emb, top_k)
    if D[0][0] < (1 - threshold):  # distance -> similarity
        return st.session_state.vector_answers[I[0][0]]
    return None

# ---------------------- FILE UPLOAD -----------------
uploaded_file = st.file_uploader("Upload your data CSV file", type=["csv"])

if uploaded_file:
    df = load_csv(uploaded_file)
    st.success("✅ File loaded successfully!")

    llm = OpenAI()
    sdf = SmartDataframe(df, config={"llm": llm, "enable_cache": False})

    # Display previous messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask a question about your data"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            placeholder = st.empty()

            # 1️⃣ Check query cache
            if prompt in st.session_state.query_cache:
                placeholder.markdown(
                    f"🔁 *Cached result:* {st.session_state.query_cache[prompt]}"
                )
                answer = st.session_state.query_cache[prompt]

            else:
                # 2️⃣ Vector search for similar queries
                similar = search_similar(prompt)
                if similar:
                    placeholder.markdown(f"🔍 *Similar result:* {similar}")
                    answer = similar
                else:
                    # 3️⃣ Try quick pandas heuristics for speed
                    try:
                        if "mean" in prompt.lower():
                            col = next(
                                (c for c in df.columns if c.lower() in prompt.lower()),
                                None,
                            )
                            if col and pd.api.types.is_numeric_dtype(df[col]):
                                answer = f"Mean of **{col}**: {df[col].mean():.2f}"
                                placeholder.markdown(answer)
                            else:
                                raise ValueError
                        else:
                            raise ValueError
                    except Exception:
                        # 4️⃣ Fallback to LLM via PandasAI
                        with st.spinner("🤖 Analyzing with LLM..."):
                            try:
                                response = sdf.chat(prompt)
                                if isinstance(response, pd.DataFrame):
                                    placeholder.dataframe(response)
                                    answer = response.to_markdown()
                                else:
                                    placeholder.markdown(str(response))
                                    answer = str(response)
                            except Exception as e:
                                answer = f"❌ Error: {e}"
                                placeholder.error(answer)

                # Update caches
                st.session_state.query_cache[prompt] = answer
                st.session_state.vector_questions.append(prompt)
                st.session_state.vector_answers.append(answer)
                build_vector_db()

            # Add assistant response to history
            st.session_state.messages.append(
                {"role": "assistant", "content": answer}
            )
# ---------------------- NO FILE ---------------------