# API Reference Guide

## Overview

This document provides detailed API documentation for all major classes and functions in the Onkar_FetiiAI project.

---

## Core Modules

### app.py - Main Application

The primary Streamlit application orchestrating the chatbot interface.

#### Global Configuration

```python
st.set_page_config(layout="wide")
```

Sets the Streamlit page to use the full browser width for better data visualization.

#### Key Variables

**`data_path`**: String
- Path to the CSV data file
- Default: `'data/Fetii_data.csv'`

**`st.session_state.messages`**: List[Dict]
- Stores chat message history
- Format: `[{"role": "user"|"assistant", "content": any}, ...]`

**`st.session_state.cache`**: Dict[str, any]
- Session-level cache for exact query matches
- Format: `{query_string: response_object}`

#### Result Type Handling

The application handles multiple result types from PandasAI:

```python
if isinstance(result, pd.DataFrame):
    st.dataframe(result, use_container_width=True)
elif isinstance(result, matplotlib.figure.Figure):
    st.pyplot(result)
elif isinstance(result, types.ModuleType) and result.__name__ == "matplotlib.pyplot":
    st.pyplot(plt.gcf())
elif isinstance(result, (str, int, float, np.floating, np.integer)):
    st.markdown(f"**Answer:** {result}")
```

---

## query_engine.py

### QueryEngine Class

Wraps PandasAI to provide natural language query processing for pandas DataFrames.

#### Constructor

```python
QueryEngine(df: pd.DataFrame)
```

**Parameters**:
- `df` (pd.DataFrame): The DataFrame to analyze

**Initializes**:
- OpenAI LLM instance
- PandasAI Config object
- SmartDataframe wrapper

**Example**:
```python
import pandas as pd
from query_engine import QueryEngine

df = pd.read_csv('data/Fetii_data.csv')
engine = QueryEngine(df)
```

#### Methods

##### answer()

```python
def answer(self, query: str) -> Union[pd.DataFrame, matplotlib.figure.Figure, str]
```

Processes a natural language query and returns the result.

**Parameters**:
- `query` (str): Natural language question about the data

**Returns**:
- `pd.DataFrame`: When query requests tabular data
- `matplotlib.figure.Figure`: When query requests a visualization
- `str`: When query requests text answer or on error

**Raises**:
- Returns error message string (doesn't raise exceptions)

**Example**:
```python
# Numeric result
result = engine.answer("What is the average age?")
# Returns: "The average age is 42.5"

# DataFrame result
result = engine.answer("Show me the top 5 destinations")
# Returns: pd.DataFrame with 5 rows

# Chart result
result = engine.answer("Create a bar chart of trips by day")
# Returns: matplotlib.figure.Figure object
```

**Internal Flow**:
1. Query passed to SmartDataframe.chat()
2. PandasAI generates Python code
3. Code executed on DataFrame
4. Result returned with appropriate type

#### Configuration

The Config object controls PandasAI behavior:

```python
Config(
    llm=self.llm,                          # LLM instance to use
    enable_cache=False,                    # Disable internal caching
    save_charts=False,                     # Don't save charts to disk
    use_error_correction_framework=False   # Disable retry logic
)
```

**Why these settings?**:
- `enable_cache=False`: Application implements custom multi-tier caching
- `save_charts=False`: Charts rendered in-memory only for security
- `use_error_correction_framework=False`: Manual error handling for better UX

---

## vector_store.py

### VectorDB Class

Implements semantic caching using sentence embeddings and FAISS vector similarity search.

#### Constructor

```python
VectorDB(model_name: str = "all-MiniLM-L6-v2")
```

**Parameters**:
- `model_name` (str, optional): Sentence-Transformers model name
  - Default: `"all-MiniLM-L6-v2"` (384-dimensional embeddings)
  - Alternatives: `"all-mpnet-base-v2"` (768-dim, more accurate but slower)

**Initializes**:
- Sentence-Transformers model for encoding
- FAISS IndexFlatL2 for L2 distance search
- Empty lists for queries and responses

**Example**:
```python
from vector_store import VectorDB

# Default model
vectordb = VectorDB()

# Custom model
vectordb = VectorDB(model_name="all-mpnet-base-v2")
```

#### Methods

##### add()

```python
def add(self, query: str, response: any) -> None
```

Adds a query-response pair to the vector cache.

**Parameters**:
- `query` (str): The user's question
- `response` (any): The answer to cache (can be any type)

**Side Effects**:
- Encodes query to 384-dimensional vector
- Adds vector to FAISS index
- Appends query and response to internal lists

**Example**:
```python
vectordb.add("What is the average age?", 42.5)
vectordb.add("Show top destinations", dataframe_result)
```

**Implementation Details**:
```python
# Encode query to float32 vector
emb = self.model.encode([query]).astype("float32")

# Add to FAISS index
self.index.add(emb)

# Store for retrieval
self.queries.append(query)
self.responses.append(response)
```

##### search()

```python
def search(self, query: str, threshold: float = 0.85) -> Optional[any]
```

Searches for a semantically similar cached query.

**Parameters**:
- `query` (str): The user's question
- `threshold` (float, optional): Similarity threshold (0.0 to 1.0)
  - Default: 0.85 (85% similarity required)
  - Higher values: More strict, fewer false positives
  - Lower values: More lenient, more cache hits

**Returns**:
- Cached response if similar query found (distance < (1 - threshold))
- `None` if no similar query or cache is empty

**Example**:
```python
# Strict matching
result = vectordb.search("What is avg age?", threshold=0.90)

# Lenient matching
result = vectordb.search("What is avg age?", threshold=0.75)
```

**Similarity Calculation**:
```python
# Encode query
emb = self.model.encode([query]).astype("float32")

# Search FAISS index for top 1 match
D, I = self.index.search(emb, 1)

# D[0][0] is L2 distance to nearest neighbor
# Convert distance to similarity score
if D[0][0] < (1 - threshold):
    return self.responses[I[0][0]]
return None
```

**Distance Interpretation**:
- L2 distance of 0.0 = identical vectors (100% similar)
- L2 distance of 0.15 = ~85% similar (default threshold)
- L2 distance of 0.30 = ~70% similar

---

## data_handler.py

### load_data()

```python
@st.cache_data(show_spinner=False)
def load_data(uploaded_file: Union[str, FileUploader]) -> pd.DataFrame
```

Loads a CSV file into a pandas DataFrame with Streamlit caching.

**Parameters**:
- `uploaded_file`: CSV file path (str) or Streamlit FileUploader object

**Returns**:
- `pd.DataFrame`: Loaded CSV data

**Caching**:
- `@st.cache_data`: Caches result to avoid reloading on every rerun
- `show_spinner=False`: Disables loading spinner

**Example**:
```python
from data_handler import load_data

# Load from path
df = load_data('data/Fetii_data.csv')

# Load from Streamlit uploader
uploaded = st.file_uploader("Upload CSV", type=['csv'])
if uploaded:
    df = load_data(uploaded)
```

---

## Alternative Implementations

### chatbot.py - LangChain RAG

This alternative implementation uses retrieval-augmented generation with LangChain.

#### load_and_preprocess_data()

```python
@st.cache_resource
def load_and_preprocess_data() -> Optional[pd.DataFrame]
```

Loads and preprocesses the Fetii dataset with date parsing.

**Returns**:
- `pd.DataFrame`: Preprocessed data with datetime 'Date' column
- `None`: On error

**Preprocessing Steps**:
1. Load CSV from `processed_merged_data_with_day.csv`
2. Convert 'Date' column to datetime
3. Drop rows with invalid dates
4. Validate DataFrame is not empty

#### build_chatbot()

```python
@st.cache_resource
def build_chatbot(df: pd.DataFrame) -> Optional[RetrievalQA]
```

Builds a RAG chatbot using LangChain components.

**Parameters**:
- `df` (pd.DataFrame): Preprocessed data

**Returns**:
- `RetrievalQA`: LangChain retrieval chain
- `None`: On error

**Architecture**:
1. Convert DataFrame rows to LangChain Documents
2. Create FAISS vector store with OpenAI embeddings
3. Set up OpenAI LLM
4. Create RetrievalQA chain with custom prompt

**Document Format**:
```python
# Each row becomes a document
content = "Date: 2023-01-01, Time: 14:30, Drop Off Address: 123 Main St, ..."
doc = Document(page_content=content)
```

**Usage**:
```python
chatbot = build_chatbot(df)
response = chatbot.invoke({"query": "What is the average age?"})
answer = response.get('result', 'No response found.')
```

### chatbot1.py - Enhanced Caching

This implementation adds a fourth resolution tier with pandas heuristics.

#### Four-Tier Resolution

```python
# Tier 1: Exact query cache
if prompt in st.session_state.query_cache:
    return cached_answer

# Tier 2: Vector semantic search
similar = search_similar(prompt)
if similar:
    return similar

# Tier 3: Pandas heuristics
if "mean" in prompt.lower():
    # Fast path for mean calculations
    return df[col].mean()

# Tier 4: LLM fallback
return sdf.chat(prompt)
```

#### search_similar()

```python
def search_similar(query: str, top_k: int = 1, threshold: float = 0.85) -> Optional[str]
```

Similar to VectorDB.search() but with configurable top_k.

**Parameters**:
- `query` (str): User's question
- `top_k` (int): Number of similar queries to retrieve
- `threshold` (float): Similarity threshold

**Returns**:
- Cached answer if found, else `None`

---

## Data Types

### Message Format

Chat messages in `st.session_state.messages`:

```python
{
    "role": str,      # "user" or "assistant"
    "content": any    # Can be str, DataFrame, Figure, etc.
}
```

### Cache Entry Format

Cache entries in `st.session_state.cache`:

```python
{
    "query_string": response_object  # Any type: str, DataFrame, Figure, etc.
}
```

### Vector Store Internal Format

```python
VectorDB.queries: List[str]           # Query strings
VectorDB.responses: List[any]         # Corresponding responses
VectorDB.index: faiss.IndexFlatL2     # FAISS vector index
```

---

## Error Handling

### QueryEngine Errors

```python
try:
    result = self.sdf.chat(query)
    return result
except Exception as e:
    return f"⚠️ Error: {e}"
```

Returns error message as string instead of raising exceptions.

### VectorDB Errors

VectorDB methods don't raise exceptions. They return `None` on failure:

```python
if len(self.queries) == 0:
    return None  # Empty cache
```

### Application-Level Errors

App displays user-friendly error messages:

```python
if not exists(data_path):
    st.error(f"Error: The data file '{data_path}' was not found.")
```

---

## Performance Considerations

### Caching Performance

| Tier | Latency | Cost | Hit Rate (typical) |
|------|---------|------|-------------------|
| Session Cache | <50ms | $0 | 30-40% |
| Vector Cache | 100-300ms | ~$0.0001 | 20-30% |
| LLM Processing | 2-8s | $0.005-0.02 | 30-50% |

### Memory Usage

- Base application: ~500 MB
- Sentence-Transformers model: ~400 MB
- FAISS index: ~0.1 MB per 100 queries
- Large DataFrame: Memory proportional to data size

### Optimization Tips

1. **Pre-populate vector cache** at startup with common queries
2. **Adjust threshold** based on accuracy vs. speed requirements
3. **Use smaller LLM** (gpt-3.5-turbo) for faster/cheaper responses
4. **Limit vector cache size** to prevent memory growth

---

## Configuration Reference

### Environment Variables

```bash
OPENAI_API_KEY=sk-...           # Required: OpenAI API key
```

### Streamlit Configuration

```python
# Page layout
st.set_page_config(layout="wide")

# Cache decorators
@st.cache_data        # For data loading (serializable)
@st.cache_resource    # For ML models (non-serializable)
```

### Model Selection

**Sentence-Transformers Models**:
- `all-MiniLM-L6-v2`: 384-dim, fast, good accuracy (default)
- `all-mpnet-base-v2`: 768-dim, slower, better accuracy
- `paraphrase-multilingual-MiniLM-L12-v2`: Multilingual support

**OpenAI Models**:
- `gpt-4`: Best accuracy, slower, more expensive
- `gpt-3.5-turbo`: Good balance (recommended)
- `gpt-3.5-turbo-instruct`: Used in chatbot.py

---

## Testing

### Unit Test Example

```python
import pandas as pd
from query_engine import QueryEngine

def test_query_engine():
    # Create test DataFrame
    df = pd.DataFrame({
        'Age': [25, 30, 35],
        'Passengers': [1, 2, 3]
    })

    # Initialize engine
    engine = QueryEngine(df)

    # Test query
    result = engine.answer("What is the average age?")
    assert isinstance(result, (str, float))
```

### Integration Test Example

```python
from vector_store import VectorDB

def test_vector_cache():
    vectordb = VectorDB()

    # Add entry
    vectordb.add("What is the average age?", 30.0)

    # Test exact match
    result = vectordb.search("What is the average age?")
    assert result == 30.0

    # Test similar query
    result = vectordb.search("What is the avg age?")
    assert result == 30.0  # Should match due to similarity

    # Test dissimilar query
    result = vectordb.search("How many trips?")
    assert result is None  # Should not match
```

---

## Debugging

### Enable Verbose Mode

```python
# In query_engine.py
self.sdf = SmartDataframe(df, config=self.config, verbose=True)
```

### Check PandasAI Logs

```bash
tail -f pandasai.log
```

### Inspect Vector Cache

```python
print(f"Cache size: {len(vectordb.queries)}")
print(f"Queries: {vectordb.queries}")
```

### Debug Session State

```python
import streamlit as st
st.write("Messages:", st.session_state.messages)
st.write("Cache:", st.session_state.cache)
```

---

**Last Updated**: March 2026
**Version**: 1.0
