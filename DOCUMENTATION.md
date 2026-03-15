# Onkar_FetiiAI - Complete Documentation

## Table of Contents
1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [Component Documentation](#component-documentation)
4. [Installation & Setup](#installation--setup)
5. [Usage Guide](#usage-guide)
6. [Technical Stack](#technical-stack)
7. [Data Flow](#data-flow)
8. [API Reference](#api-reference)
9. [Configuration](#configuration)
10. [Alternative Implementations](#alternative-implementations)
11. [Performance Optimization](#performance-optimization)
12. [Troubleshooting](#troubleshooting)

---

## Project Overview

**Onkar_FetiiAI** is an intelligent, conversational data analysis platform built with Streamlit that enables users to interact with CSV datasets using natural language queries. The application transforms complex data analysis tasks into simple conversations, making data insights accessible to both technical and non-technical users.

### Key Features

- **Natural Language Querying**: Ask questions in plain English instead of writing SQL or pandas code
- **Multi-Level Caching**: Three-tier caching system (session cache, vector semantic cache, LLM processing) for optimal performance
- **Dynamic Result Rendering**: Automatically displays results as tables, charts, or text based on query type
- **Semantic Search**: Uses vector embeddings to find similar previously-answered queries
- **Multiple Implementations**: Three different chatbot approaches (PandasAI, LangChain RAG, Enhanced multi-cache)

### Use Case

The primary use case is analyzing Fetii rideshare trip data, including:
- Passenger demographics and travel patterns
- Time-based analysis (peak hours, popular days)
- Geographic insights (popular destinations)
- Group travel behavior analysis

---

## Architecture

### High-Level System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Streamlit Web UI                        │
│                         (app.py)                             │
└──────────────┬──────────────────────────────────────────────┘
               │
               │ User Query
               ▼
┌─────────────────────────────────────────────────────────────┐
│              Three-Tier Query Resolution                     │
├─────────────────────────────────────────────────────────────┤
│  Tier 1: Session Cache (Exact Match)                        │
│  └─── O(1) lookup in st.session_state.cache                 │
│                                                              │
│  Tier 2: Vector Semantic Cache                              │
│  └─── Sentence-Transformers + FAISS (vector_store.py)       │
│       └─── Returns cached answer if similarity > 85%        │
│                                                              │
│  Tier 3: LLM Processing                                     │
│  └─── QueryEngine (query_engine.py)                         │
│       └─── PandasAI + OpenAI GPT                            │
│            └─── Converts NL → Python → Result               │
└─────────────────────────────────────────────────────────────┘
               │
               │ Result
               ▼
┌─────────────────────────────────────────────────────────────┐
│              Result Type Handler                             │
├─────────────────────────────────────────────────────────────┤
│  • pandas.DataFrame    → Interactive table                   │
│  • matplotlib.Figure   → Chart/visualization                 │
│  • String/Number       → Formatted text                      │
│  • Module (pyplot)     → Special pyplot handling             │
└─────────────────────────────────────────────────────────────┘
```

### Directory Structure

```
Onkar_FetiiAI/
├── app.py                    # Main application entry point
├── query_engine.py           # LLM-powered query processor
├── vector_store.py           # Semantic caching with FAISS
├── data_handler.py           # CSV data loading utilities
├── chatbot.py                # Alternative: LangChain RAG implementation
├── chatbot1.py               # Alternative: Enhanced multi-cache system
├── testing.py                # Data analysis utilities
├── data/
│   ├── __init__.py
│   └── Fetii_data.csv        # Rideshare trip dataset (~420 KB)
├── requirements.txt          # Python dependencies
├── README.md                 # Basic project information
├── DOCUMENTATION.md          # This comprehensive documentation
└── .gitignore               # Git ignore configuration
```

---

## Component Documentation

### 1. app.py - Main Application

**Purpose**: Orchestrates the entire application, managing UI, state, and query flow.

**Key Responsibilities**:
- Streamlit UI initialization and rendering
- Chat message history management
- Three-tier cache coordination
- Dynamic result type handling
- User interaction flow control

**Code Flow**:
```python
1. Load data from CSV (data/Fetii_data.csv)
2. Initialize QueryEngine and VectorDB
3. Initialize session state (messages, cache)
4. Display chat history
5. Handle new user input:
   a. Check session cache (Tier 1)
   b. Check vector semantic cache (Tier 2)
   c. Call LLM via QueryEngine (Tier 3)
6. Store result in caches
7. Display result based on type
8. Update chat history
```

**Important Functions**:
- Session state management: `st.session_state.messages` and `st.session_state.cache`
- Result rendering: Type-based display logic (lines 72-81)

### 2. query_engine.py - LLM Query Processing

**Purpose**: Wraps PandasAI to convert natural language queries into executable DataFrame operations.

**Key Components**:
- `QueryEngine` class: Main interface for query processing
- `SmartDataframe`: PandasAI's intelligent DataFrame wrapper
- OpenAI LLM integration

**Configuration**:
```python
Config(
    llm=OpenAI(),
    enable_cache=False,      # Disabled (app-level caching used instead)
    save_charts=False,       # Charts not saved to disk
    use_error_correction_framework=False  # Direct execution
)
```

**API**:
- `QueryEngine.__init__(df: pd.DataFrame)`: Initialize with DataFrame
- `QueryEngine.answer(query: str)`: Process query and return result
  - Returns: `pd.DataFrame`, `matplotlib.Figure`, or `str`
  - Error handling: Returns error message string on exception

### 3. vector_store.py - Semantic Cache

**Purpose**: Implements vector-based semantic search to retrieve cached answers for similar queries.

**How It Works**:
1. Encode queries using Sentence-Transformers (`all-MiniLM-L6-v2`)
2. Store 384-dimensional embeddings in FAISS index
3. For new queries, compute similarity to cached queries
4. Return cached answer if L2 distance suggests high similarity

**Key Methods**:
- `VectorDB.__init__(model_name="all-MiniLM-L6-v2")`: Initialize embedder and FAISS index
- `VectorDB.add(query, response)`: Add query-response pair to cache
- `VectorDB.search(query, threshold=0.85)`: Search for similar cached query
  - Returns cached response if similarity > 85%, else `None`

**Similarity Calculation**:
```python
# L2 distance approximates cosine similarity for normalized vectors
if D[0][0] < (1 - threshold):  # threshold=0.85 by default
    return cached_response
```

### 4. data_handler.py - Data Loading

**Purpose**: Loads CSV files into pandas DataFrames with Streamlit caching.

**Features**:
- `@st.cache_data` decorator: Prevents redundant file reads
- Simple CSV loading with error handling

**API**:
- `load_data(uploaded_file)`: Loads CSV and caches result

### 5. Alternative Implementations

#### chatbot.py - LangChain RAG Approach

**Purpose**: Document-based retrieval-augmented generation using LangChain.

**Architecture**:
- Converts each CSV row into a LangChain `Document`
- Creates FAISS vector store with OpenAI embeddings
- Uses `RetrievalQA` chain for question answering
- Custom prompt template for data analysis context

**Key Differences from app.py**:
- Row-wise document chunking (vs. DataFrame-level processing)
- OpenAI embeddings (vs. Sentence-Transformers)
- Different data file: `processed_merged_data_with_day.csv`

#### chatbot1.py - Enhanced Multi-Cache System

**Purpose**: Advanced implementation with four-tier resolution strategy.

**Resolution Pipeline**:
1. **Exact query cache**: Direct dictionary lookup
2. **Vector semantic search**: FAISS-based similarity
3. **Pandas heuristics**: Fast path for simple operations (e.g., "mean")
4. **LLM fallback**: Full PandasAI processing

**Unique Features**:
- File upload widget (vs. fixed file path in app.py)
- Heuristic-based shortcuts for common queries
- More aggressive caching strategy

---

## Installation & Setup

### Prerequisites

- Python 3.8+ (tested on Python 3.9+)
- pip package manager
- OpenAI API key

### Step 1: Clone Repository

```bash
git clone https://github.com/onkar2002406/Onkar_FetiiAI.git
cd Onkar_FetiiAI
```

### Step 2: Create Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**Note**: The requirements.txt includes 200+ dependencies. Key libraries:
- streamlit==1.49.1
- pandasai==2.3.2
- openai==1.108.1
- sentence-transformers==5.1.0
- faiss-cpu==1.12.0
- pandas==1.5.3

### Step 4: Configure API Keys

Create a `.env` file in the project root:

```bash
# .env
OPENAI_API_KEY=your_openai_api_key_here
```

**Alternative**: Set as environment variable:
```bash
export OPENAI_API_KEY="your_openai_api_key_here"
```

### Step 5: Verify Data File

Ensure `data/Fetii_data.csv` exists:
```bash
ls data/Fetii_data.csv
```

### Step 6: Run the Application

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

---

## Usage Guide

### Starting a Chat Session

1. Launch the app: `streamlit run app.py`
2. Wait for "Data loaded successfully!" confirmation
3. Type your question in the chat input at the bottom
4. Press Enter to submit

### Example Queries

**Basic Statistics**:
```
What is the average age of passengers?
How many total trips are in the dataset?
What is the most common number of passengers?
```

**Time-Based Analysis**:
```
What are the busiest hours for trips?
Show me trips by day of week
How many trips happened on Mondays?
```

**Demographic Insights**:
```
What is the age distribution of passengers?
Show me trips by age group
How many trips had 18-24 year old passengers?
```

**Geographic Analysis**:
```
What are the top 10 drop-off locations?
Where do people travel most on weekends?
Show me the most popular destinations
```

**Visualizations**:
```
Create a bar chart of trips by day
Plot passenger count distribution
Show a histogram of trip times
```

### Understanding Results

- **Tables**: Displayed as interactive DataFrames (sortable, scrollable)
- **Charts**: Rendered as matplotlib figures (downloadable)
- **Text**: Formatted answers with numeric results

### Performance Tips

1. **Repeated Queries**: Exact same questions return instantly (cached)
2. **Similar Queries**: Semantically similar questions return quickly (vector cache)
3. **Complex Analysis**: First-time complex queries may take 3-10 seconds (LLM processing)

---

## Technical Stack

### Frontend & UI
| Library | Version | Purpose |
|---------|---------|---------|
| Streamlit | 1.49.1 | Web UI framework |
| Matplotlib | 3.10.6 | Chart visualization |

### Data Processing
| Library | Version | Purpose |
|---------|---------|---------|
| Pandas | 1.5.3 | DataFrame operations |
| NumPy | 1.26.4 | Numerical computing |
| DuckDB | 1.4.0 | SQL operations |

### Natural Language & LLM
| Library | Version | Purpose |
|---------|---------|---------|
| PandasAI | 2.3.2 | NL → DataFrame queries |
| OpenAI | 1.108.1 | GPT language models |
| LangChain | (community) | RAG pipelines |

### Vector Search & Embeddings
| Library | Version | Purpose |
|---------|---------|---------|
| Sentence-Transformers | 5.1.0 | Text embeddings |
| FAISS | 1.12.0 | Vector similarity search |
| Scikit-learn | 1.7.2 | ML utilities |

### Alternative LLM Providers (Available)
- Azure OpenAI
- Mistral AI (1.9.10)
- Cohere (5.18.0)
- LLaMA Index (0.14.0)

---

## Data Flow

### Query Processing Flow

```
User Query: "What is the average age?"
│
▼
┌─────────────────────────────┐
│ Tier 1: Session Cache       │
│ Check: prompt in cache dict │
└─────────────┬───────────────┘
              │ NOT FOUND
              ▼
┌─────────────────────────────────────┐
│ Tier 2: Vector Semantic Search      │
│ 1. Encode query with Sentence-BERT  │
│ 2. Search FAISS index                │
│ 3. Compare L2 distance               │
└─────────────┬───────────────────────┘
              │ NOT FOUND (distance > threshold)
              ▼
┌─────────────────────────────────────┐
│ Tier 3: LLM Processing              │
│ 1. PandasAI receives query           │
│ 2. GPT generates Python code:        │
│    df['Age'].mean()                  │
│ 3. Execute code on DataFrame         │
│ 4. Return result: 42.5               │
└─────────────┬───────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│ Store in Caches                      │
│ • session_state.cache[query] = 42.5  │
│ • vectordb.add(query, 42.5)          │
└─────────────┬───────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│ Display Result                       │
│ "Answer: 42.5"                       │
└─────────────────────────────────────┘
```

### Caching Strategy

**Why Three Tiers?**
1. **Session Cache**: Instant response for repeated questions in same session
2. **Vector Cache**: Fast response for semantically similar questions
3. **LLM Processing**: Full intelligence for novel questions

**Cost Optimization**:
- Tier 1: $0.00 per query (in-memory)
- Tier 2: ~$0.0001 per query (embedding computation)
- Tier 3: ~$0.005-0.02 per query (OpenAI API call)

---

## API Reference

### QueryEngine Class

```python
class QueryEngine:
    def __init__(self, df: pd.DataFrame)
```
Initialize the query engine with a DataFrame.

**Parameters**:
- `df`: pandas DataFrame containing the data to analyze

**Example**:
```python
from query_engine import QueryEngine
import pandas as pd

df = pd.read_csv('data/Fetii_data.csv')
engine = QueryEngine(df)
```

---

```python
def answer(self, query: str) -> Union[pd.DataFrame, matplotlib.figure.Figure, str]
```
Process a natural language query and return the result.

**Parameters**:
- `query`: Natural language question about the data

**Returns**:
- `pd.DataFrame`: Tabular results
- `matplotlib.figure.Figure`: Visualizations
- `str`: Text answers or error messages

**Example**:
```python
result = engine.answer("What is the average passenger count?")
# Returns: "The average passenger count is 3.2"
```

### VectorDB Class

```python
class VectorDB:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2")
```
Initialize the vector database for semantic caching.

**Parameters**:
- `model_name`: Sentence-Transformers model name (default: "all-MiniLM-L6-v2")

---

```python
def add(self, query: str, response: Any) -> None
```
Add a query-response pair to the cache.

**Parameters**:
- `query`: The user's question
- `response`: The answer to cache

---

```python
def search(self, query: str, threshold: float = 0.85) -> Optional[Any]
```
Search for a similar cached query.

**Parameters**:
- `query`: The user's question
- `threshold`: Similarity threshold (0-1, default: 0.85)

**Returns**:
- Cached response if similar query found, else `None`

---

## Configuration

### PandasAI Configuration

Located in `query_engine.py`:

```python
Config(
    llm=self.llm,                          # OpenAI LLM instance
    enable_cache=False,                    # Disable internal caching
    save_charts=False,                     # Don't save charts to disk
    use_error_correction_framework=False   # Disable auto-retry
)
```

**Why These Settings?**
- `enable_cache=False`: App implements custom multi-tier caching
- `save_charts=False`: Charts rendered in-memory only
- `use_error_correction_framework=False`: Faster execution, manual error handling

### Vector Search Configuration

Located in `vector_store.py`:

```python
VectorDB(model_name="all-MiniLM-L6-v2")  # 384-dimensional embeddings
```

**Tunable Parameters**:
- `threshold=0.85`: Similarity threshold in `search()` method
  - Higher (0.9+): More strict, fewer false positives
  - Lower (0.7-0.8): More lenient, more cache hits

### Streamlit Configuration

Page layout in `app.py`:
```python
st.set_page_config(layout="wide")  # Use full browser width
```

---

## Alternative Implementations

### When to Use Each Version

| Implementation | Use Case | Strengths | Limitations |
|----------------|----------|-----------|-------------|
| **app.py** | Production, general use | Simple, reliable, handles multiple result types | Fixed file path |
| **chatbot.py** | Document-heavy datasets | Row-level retrieval, LangChain ecosystem | Different file requirement |
| **chatbot1.py** | High query volume | 4-tier caching, heuristics, file upload | More complex, harder to debug |

### Migration Guide

**From app.py to chatbot1.py**:
1. Change data loading to use file uploader
2. Implement heuristic shortcuts for common queries
3. Add 4th tier (pandas heuristics) to resolution

**From chatbot.py to app.py**:
1. Remove LangChain dependencies
2. Switch from row-wise documents to DataFrame-level processing
3. Update data file path to `data/Fetii_data.csv`

---

## Performance Optimization

### Caching Effectiveness

**Expected Cache Hit Rates**:
- Session Cache (Tier 1): 30-40% of queries
- Vector Cache (Tier 2): 20-30% of queries
- LLM Calls (Tier 3): 30-50% of queries

### Optimization Strategies

1. **Pre-populate Vector Cache**:
```python
# Add common queries at startup
common_queries = [
    "What is the average age?",
    "How many trips are there?",
    "What are the top destinations?"
]
for q in common_queries:
    ans = engine.answer(q)
    vectordb.add(q, ans)
```

2. **Adjust Vector Similarity Threshold**:
```python
# More aggressive caching
vectordb.search(prompt, threshold=0.80)  # Down from 0.85
```

3. **Use Smaller LLM for Simple Queries**:
```python
# In query_engine.py
self.llm = OpenAI(model="gpt-3.5-turbo")  # Faster, cheaper
```

### Performance Benchmarks

| Query Type | Tier 1 | Tier 2 | Tier 3 |
|------------|--------|--------|--------|
| Exact match | <50ms | - | - |
| Similar query | - | 100-300ms | - |
| Novel query | - | - | 2-8s |

---

## Troubleshooting

### Common Issues

**Issue**: "Error: The data file 'data/Fetii_data.csv' was not found"
- **Solution**: Verify file exists at `/path/to/project/data/Fetii_data.csv`
- Check working directory: `pwd` in terminal

**Issue**: OpenAI API errors
- **Solution**: Verify API key in `.env` file
- Check key validity: `echo $OPENAI_API_KEY`
- Ensure billing is active on OpenAI account

**Issue**: FAISS installation errors
- **Solution**: Use CPU version: `pip install faiss-cpu`
- For GPU: `pip install faiss-gpu` (requires CUDA)

**Issue**: Slow first query
- **Expected**: First query loads models (Sentence-Transformers) into memory
- **Solution**: Add warmup query at startup

**Issue**: "Unrecognized content type" message
- **Cause**: PandasAI returned unexpected type
- **Solution**: Check `app.py` lines 40-48 for type handling

### Debug Mode

Enable verbose logging:
```python
# In query_engine.py
self.sdf = SmartDataframe(df, config=self.config, verbose=True)
```

Check PandasAI logs:
```bash
tail -f pandasai.log
```

### Resource Usage

**Memory**:
- Base app: ~500 MB
- With models loaded: ~2 GB
- Large datasets: +1 GB per 1M rows

**Disk Space**:
- Dependencies: ~2 GB
- Model cache: ~500 MB (Sentence-Transformers)

---

## Contributing

### Code Style

- Follow existing naming conventions (descriptive full-word names)
- Use type hints where applicable
- Add docstrings to all functions
- Keep functions focused and single-purpose

### Memory from Previous Sessions

Key conventions:
- Use descriptive variable names: `answer` not `ans`, `message` not `m`, `embedding` not `emb`
- Use `distances, indices` instead of `D, I` for FAISS results
- Source: `app.py:37-48`, `query_engine.py:21`, `vector_store.py:45-62`

### Adding New Features

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Implement changes with tests
4. Update documentation
5. Submit pull request

---

## Future Enhancements

### Planned Features
- [ ] Multi-file upload support
- [ ] Export analysis results (PDF, Excel)
- [ ] Custom visualization templates
- [ ] User authentication and session persistence
- [ ] Real-time data streaming support

### Extensibility Points
- **Alternative LLMs**: Swap OpenAI for Azure, Mistral, or Cohere
- **Vector Databases**: Replace FAISS with Pinecone or Weaviate
- **Data Sources**: Add SQL, API, or real-time data connectors
- **Deployment**: Package as Docker container or deploy to Streamlit Cloud

---

## License

See repository for license information.

## Contact

For questions or issues, please open a GitHub issue at the repository.

---

**Last Updated**: March 2026
**Version**: 1.0
**Maintained By**: Onkar (onkar2002406)
