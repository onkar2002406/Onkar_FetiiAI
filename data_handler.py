import pandas as pd
import streamlit as st

@st.cache_data(show_spinner=False)
def load_data(uploaded_file):
    """Loads a CSV file into a pandas DataFrame and caches it."""
    return pd.read_csv(uploaded_file)