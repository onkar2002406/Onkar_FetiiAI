# Fetii Data Chatbot 🚖

An intelligent conversational interface for analyzing rideshare trip data using natural language. Built with Streamlit, PandasAI, and powered by advanced caching mechanisms for optimal performance.

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.49.1-FF4B4B.svg)](https://streamlit.io/)
[![PandasAI](https://img.shields.io/badge/pandasai-2.3.2-green.svg)](https://github.com/gventuri/pandas-ai)

## 🌟 Features

- **Natural Language Queries**: Ask questions about your data in plain English
- **Smart Caching System**: Three-tier caching (session, semantic vector, LLM) for fast responses
- **Dynamic Result Rendering**: Automatically displays tables, charts, or text based on query results
- **Semantic Search**: Finds similar previously-answered questions using vector embeddings
- **Real-time Analysis**: Instant data insights without writing code

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- OpenAI API key

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/onkar2002406/Onkar_FetiiAI.git
   cd Onkar_FetiiAI
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up your OpenAI API key**

   Create a `.env` file in the project root:
   ```bash
   OPENAI_API_KEY=your_api_key_here
   ```

5. **Run the application**
   ```bash
   streamlit run app.py
   ```

The app will open automatically in your browser at `http://localhost:8501`.

## 💡 Usage Examples

Once the app is running, you can ask questions like:

**Basic Statistics**:
- "What is the average age of passengers?"
- "How many trips are in the dataset?"
- "What is the most common drop-off location?"

**Time-Based Analysis**:
- "What are the busiest hours for trips?"
- "Show me trips by day of week"
- "How many trips happened on Monday evenings?"

**Demographic Insights**:
- "What is the age distribution of passengers?"
- "How many trips had 18-24 year old passengers?"

**Visualizations**:
- "Create a bar chart of trips by day"
- "Plot the passenger count distribution"
- "Show a histogram of trip times"

## 🏗️ Architecture

The application uses a sophisticated three-tier query resolution system:

```
User Query
    ↓
┌─────────────────────────────┐
│ Tier 1: Session Cache       │  ← Instant (exact match)
│ (in-memory dictionary)      │
└─────────────┬───────────────┘
              ↓ (if not found)
┌─────────────────────────────┐
│ Tier 2: Vector Semantic     │  ← Fast (~200ms)
│ (FAISS + Sentence-BERT)     │     Finds similar queries
└─────────────┬───────────────┘
              ↓ (if not found)
┌─────────────────────────────┐
│ Tier 3: LLM Processing      │  ← Intelligent (2-8s)
│ (PandasAI + OpenAI GPT)     │     Full natural language processing
└─────────────────────────────┘
```

This design optimizes for:
- **Speed**: Most queries return in <300ms via caching
- **Cost**: Reduces expensive LLM API calls by 60-70%
- **Accuracy**: Falls back to full LLM processing when needed

## 📁 Project Structure

```
Onkar_FetiiAI/
├── app.py                # Main Streamlit application
├── query_engine.py       # PandasAI wrapper for LLM queries
├── vector_store.py       # Semantic caching with FAISS
├── data_handler.py       # CSV data loading utilities
├── chatbot.py            # Alternative: LangChain RAG implementation
├── chatbot1.py           # Alternative: Enhanced caching system
├── testing.py            # Data analysis utilities
├── data/
│   └── Fetii_data.csv    # Rideshare trip dataset
├── requirements.txt      # Python dependencies
├── DOCUMENTATION.md      # Comprehensive documentation
└── README.md            # This file
```

## 🛠️ Tech Stack

| Category | Technology | Purpose |
|----------|-----------|---------|
| **Frontend** | Streamlit | Interactive web interface |
| **Data Processing** | Pandas | DataFrame operations |
| **NL Processing** | PandasAI | Natural language to code |
| **LLM** | OpenAI GPT | Language understanding |
| **Vector Search** | FAISS | Fast similarity search |
| **Embeddings** | Sentence-Transformers | Query encoding |
| **Visualization** | Matplotlib | Chart generation |

## 📊 Dataset

The application uses the **Fetii rideshare trip dataset** containing:
- Trip dates and times
- Passenger demographics (age)
- Drop-off locations
- Group size (total passengers)
- Day of week information

**Note**: For alternative implementations (chatbot.py), you may need a different dataset file.

## 🔧 Configuration

### Environment Variables

```bash
OPENAI_API_KEY=your_api_key_here
```

### Customization Options

**Adjust semantic similarity threshold** (in `app.py`):
```python
ans = vectordb.search(prompt, threshold=0.85)  # 0.80-0.95 recommended
```

**Change LLM model** (in `query_engine.py`):
```python
self.llm = OpenAI(model="gpt-4")  # Or gpt-3.5-turbo for faster/cheaper
```

**Modify embedding model** (in `vector_store.py`):
```python
VectorDB(model_name="all-MiniLM-L6-v2")  # Or larger models for better accuracy
```

## 📖 Documentation

For comprehensive documentation including:
- Detailed architecture diagrams
- API reference
- Performance optimization guide
- Troubleshooting tips
- Alternative implementations

See [DOCUMENTATION.md](DOCUMENTATION.md)

## 🐛 Troubleshooting

**Issue**: Data file not found
- Ensure `data/Fetii_data.csv` exists in the correct location

**Issue**: OpenAI API errors
- Verify your API key is set correctly in `.env`
- Check your OpenAI account has available credits

**Issue**: Slow first query
- First query loads ML models into memory (expected)
- Subsequent queries will be much faster

**Issue**: Module import errors
- Reinstall dependencies: `pip install -r requirements.txt`
- Ensure virtual environment is activated

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 Code Style

- Use descriptive variable names (e.g., `answer` not `ans`, `message` not `m`)
- Add docstrings to functions
- Follow existing patterns in the codebase

## 🔮 Future Enhancements

- [ ] Multi-file upload support
- [ ] Export results to PDF/Excel
- [ ] User authentication
- [ ] Real-time data streaming
- [ ] Custom visualization templates
- [ ] Support for SQL databases

## 📜 License

See repository for license information.

## 👤 Author

**Onkar** (onkar2002406)

## 🙏 Acknowledgments

- Built with [Streamlit](https://streamlit.io/)
- Powered by [PandasAI](https://github.com/gventuri/pandas-ai)
- Vector search via [FAISS](https://github.com/facebookresearch/faiss)
- Embeddings from [Sentence-Transformers](https://www.sbert.net/)

---

**⭐ If you find this project useful, please consider giving it a star!**
