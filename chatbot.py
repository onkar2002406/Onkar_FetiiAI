import pandas as pd
import streamlit as st
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document

# Load environment variables for the API key
from dotenv import load_dotenv
load_dotenv()

@st.cache_resource
def load_and_preprocess_data():
    """
    Loads and preprocesses the Fetii dataset from a CSV file.
    This function is cached to run only once per session.
    """
    try:
        # Load the data, assuming the file is available
        df = pd.read_csv('processed_merged_data_with_day.csv')
        
        # Check if the 'Date' column exists. If not, the file format is unexpected.
        if 'Date' not in df.columns:
            st.error("Error: The 'Date' column was not found in the CSV file. Please ensure the file has the correct columns.")
            return None

        # Convert the 'Date' column to datetime objects
        # Pandas can automatically infer the 'YYYY-MM-DD' format.
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

        # Filter out rows with invalid dates
        df.dropna(subset=['Date'], inplace=True)

        # Check if the DataFrame is empty after dropping invalid dates.
        if df.empty:
            st.error("Error: The 'Date' column could not be parsed. Please ensure the dates are in a valid format in your CSV file.")
            return None

        return df
    
    except FileNotFoundError as e:
        st.error(f"Error: {e}. Please ensure 'processed_merged_data.csv' is in the same directory.")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred during data loading: {e}")
        return None

@st.cache_resource
def build_chatbot(df):
    """
    Builds a Retrieval-Augmented Generation (RAG) chatbot using LangChain.
    The data is processed row by row to create individual documents for the vector store.
    This function is cached to run only once per session.
    """
    if df is None:
        return None

    # Step 1: Create a LangChain Document for each row of the DataFrame
    docs = []
    # Create a simple list of column headers to be used in each document
    # This helps the LLM understand the structure of the data
    headers = df.columns.tolist()
    
    # Iterate through each row and create a document
    for _, row in df.iterrows():
        # Combine the headers and row values into a single string
        # This creates a "row-wise" document as requested
        content_string = ", ".join([f"{header}: {value}" for header, value in zip(headers, row)])
        docs.append(Document(page_content=content_string))

    st.info(f"Created {len(docs)} document chunks from the dataset rows.")

    # Step 2: Create a vector store from the documents using an OpenAI embedding model
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = FAISS.from_documents(docs, embedding_model)
    st.info("Vector store created.")

    # Step 3: Set up the LLM and the RAG chain
    llm = OpenAI(model="gpt-3.5-turbo-instruct", temperature=0.7)
    
    # Define a prompt template to guide the LLM's response
    template = """
        You are an expert data analyst chatbot. Your task is to analyze the provided trip data and answer questions with a high degree of accuracy.

        ### Data Description
        The data is provided in a row-by-row format from a CSV file.
        - The date format is MM/DD/YYYY.
        - 'Total Pass' is the number of passengers for that trip.
        - If a question cannot be answered from the data, you must explicitly state that the information is not available in the provided context.

        ### Instructions
        1.  **Analyze**: Carefully read the provided data context and the user's question.
        2.  **Calculate**: Perform all necessary calculations and data aggregations to derive the answer.
        3.  **Format**: Present the final answer clearly and concisely. If the question asks for a list, use a bulleted list. If it requires a summary, provide a paragraph.
        4.  **Constraints**: You must not invent information. All answers must be directly supported by the provided data.

        ### Context
        {context}

        ### Question
        {question}

        ### Answer
    """
    
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vector_store.as_retriever(),
        chain_type="stuff",
        verbose=True,
        chain_type_kwargs={"prompt": PromptTemplate.from_template(template)}
    )

    return rag_chain

# Streamlit UI
st.title("Fetii Data Chatbot")
st.write("Ask me questions about the Fetii trip, rider, and demographic data!")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Load data and build chatbot when the app starts
with st.spinner('Loading data and building the chatbot...'):
    df = load_and_preprocess_data()
    chatbot = build_chatbot(df)

if chatbot:
    # Accept user input
    if prompt := st.chat_input("What would you like to know?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
            
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            with st.spinner("Thinking..."):
                try:
                    # Invoke the chatbot with the user's prompt
                    response = chatbot.invoke({"query": prompt})
                    full_response = response.get('result', 'No response found.')
                except Exception as e:
                    full_response = f"An error occurred: {e}"
            
            message_placeholder.markdown(full_response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})
else:
    st.error("Chatbot could not be initialized. Please check the console for errors and ensure your API key and data files are correct.")


