# Setup and Installation Guide

## Table of Contents
1. [System Requirements](#system-requirements)
2. [Installation Methods](#installation-methods)
3. [Configuration](#configuration)
4. [Verification](#verification)
5. [Common Issues](#common-issues)
6. [Advanced Setup](#advanced-setup)

---

## System Requirements

### Hardware Requirements

**Minimum**:
- CPU: 2 cores
- RAM: 4 GB
- Disk Space: 5 GB

**Recommended**:
- CPU: 4+ cores
- RAM: 8 GB
- Disk Space: 10 GB

### Software Requirements

- **Operating System**:
  - Linux (Ubuntu 18.04+, CentOS 7+)
  - macOS 10.14+
  - Windows 10/11

- **Python**: 3.8, 3.9, 3.10, or 3.11
  - Check version: `python --version` or `python3 --version`

- **pip**: 20.0 or higher
  - Check version: `pip --version`

- **Git**: 2.0 or higher
  - Check version: `git --version`

---

## Installation Methods

### Method 1: Standard Installation (Recommended)

#### Step 1: Install Python

**Linux (Ubuntu/Debian)**:
```bash
sudo apt update
sudo apt install python3.9 python3.9-venv python3-pip
```

**macOS** (using Homebrew):
```bash
brew install python@3.9
```

**Windows**:
1. Download Python from https://www.python.org/downloads/
2. Run installer and check "Add Python to PATH"
3. Verify: `python --version`

#### Step 2: Clone the Repository

```bash
# Clone via HTTPS
git clone https://github.com/onkar2002406/Onkar_FetiiAI.git

# Or clone via SSH (if you have SSH keys set up)
git clone git@github.com:onkar2002406/Onkar_FetiiAI.git

# Navigate to project directory
cd Onkar_FetiiAI
```

#### Step 3: Create Virtual Environment

**Why use a virtual environment?**
- Isolates project dependencies
- Prevents conflicts with system Python packages
- Makes the project portable

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On Linux/macOS:
source venv/bin/activate

# On Windows (Command Prompt):
venv\Scripts\activate.bat

# On Windows (PowerShell):
venv\Scripts\Activate.ps1
```

**Verify activation**: Your prompt should show `(venv)` prefix.

#### Step 4: Upgrade pip (Optional but Recommended)

```bash
pip install --upgrade pip
```

#### Step 5: Install Dependencies

```bash
# Install all requirements
pip install -r requirements.txt

# This will install 200+ packages, taking 5-10 minutes
# Expected size: ~2 GB of downloads
```

**Note**: If you encounter issues, see [Common Issues](#common-issues) section.

#### Step 6: Configure Environment

Create a `.env` file in the project root:

```bash
# Create .env file
touch .env

# Edit with your preferred editor
nano .env  # or vim, code, notepad, etc.
```

Add your OpenAI API key:
```bash
OPENAI_API_KEY=sk-your-api-key-here
```

**How to get an OpenAI API key**:
1. Visit https://platform.openai.com/
2. Sign up or log in
3. Go to API Keys section
4. Create new secret key
5. Copy and paste into `.env` file

#### Step 7: Verify Data File

Ensure the dataset exists:
```bash
ls -lh data/Fetii_data.csv
```

Expected output: `~420 KB` file

#### Step 8: Run the Application

```bash
streamlit run app.py
```

The application will:
1. Start the Streamlit server
2. Load the dataset
3. Initialize models (first run takes ~30 seconds)
4. Open in your default browser at http://localhost:8501

---

### Method 2: Docker Installation (Advanced)

#### Prerequisites
- Docker installed: https://docs.docker.com/get-docker/
- Docker Compose (optional)

#### Create Dockerfile

```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Run the application
CMD ["streamlit", "run", "app.py", "--server.address", "0.0.0.0"]
```

#### Build and Run

```bash
# Build Docker image
docker build -t onkar-fetii-ai .

# Run container
docker run -p 8501:8501 -e OPENAI_API_KEY=your-key-here onkar-fetii-ai

# Or with .env file
docker run -p 8501:8501 --env-file .env onkar-fetii-ai
```

Access at: http://localhost:8501

---

### Method 3: Cloud Deployment (Streamlit Cloud)

#### Prerequisites
- GitHub account
- Streamlit Cloud account (free): https://streamlit.io/cloud

#### Steps

1. **Fork the repository** on GitHub

2. **Go to Streamlit Cloud**:
   - Visit https://share.streamlit.io/
   - Sign in with GitHub

3. **Deploy new app**:
   - Click "New app"
   - Select your forked repository
   - Set main file: `app.py`
   - Advanced settings → Add secrets:
     ```
     OPENAI_API_KEY = "your-api-key-here"
     ```

4. **Deploy**: Click "Deploy"

Your app will be live at: `https://your-app-name.streamlit.app`

---

## Configuration

### Environment Variables

Create `.env` file with the following variables:

```bash
# Required
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxx

# Optional - Alternative LLM Providers
AZURE_OPENAI_KEY=your-azure-key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
MISTRAL_API_KEY=your-mistral-key
COHERE_API_KEY=your-cohere-key

# Optional - Configuration
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=localhost
```

### Streamlit Configuration

Create `.streamlit/config.toml` for advanced settings:

```bash
mkdir -p .streamlit
nano .streamlit/config.toml
```

Add configurations:

```toml
[server]
port = 8501
enableCORS = false
enableXsrfProtection = true

[browser]
gatherUsageStats = false

[theme]
primaryColor = "#FF4B4B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
font = "sans serif"
```

### Application Configuration

Edit configuration directly in Python files:

**Change LLM Model** (`query_engine.py`):
```python
# Line 10: Change to GPT-4 for better accuracy
self.llm = OpenAI(model="gpt-4")

# Or GPT-3.5 Turbo for faster/cheaper
self.llm = OpenAI(model="gpt-3.5-turbo")
```

**Adjust Vector Similarity** (`app.py`):
```python
# Line 65: Change threshold (0.75-0.95)
ans = vectordb.search(prompt, threshold=0.85)
```

**Change Embedding Model** (`vector_store.py`):
```python
# Line 33: Use larger model for better accuracy
VectorDB(model_name="all-mpnet-base-v2")
```

---

## Verification

### Test Installation

After installation, verify everything works:

#### 1. Check Python Environment

```bash
which python  # Should show venv path
python --version  # Should be 3.8+
```

#### 2. Verify Packages

```bash
pip list | grep -E "streamlit|pandasai|openai"
```

Expected output:
```
openai                  1.108.1
pandasai                2.3.2
streamlit               1.49.1
```

#### 3. Test Import

```bash
python -c "import streamlit; import pandasai; print('Success')"
```

Should print: `Success`

#### 4. Test OpenAI Connection

```python
python << EOF
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Test"}],
    max_tokens=5
)
print("OpenAI connection successful!")
EOF
```

#### 5. Run Health Check

Create `health_check.py`:
```python
import pandas as pd
from data_handler import load_data
from query_engine import QueryEngine
from vector_store import VectorDB

print("1. Testing data loading...")
df = load_data('data/Fetii_data.csv')
print(f"   ✓ Loaded {len(df)} rows")

print("2. Testing QueryEngine...")
engine = QueryEngine(df)
print("   ✓ QueryEngine initialized")

print("3. Testing VectorDB...")
vectordb = VectorDB()
print("   ✓ VectorDB initialized")

print("\n✅ All components working!")
```

Run: `python health_check.py`

---

## Common Issues

### Issue 1: FAISS Installation Fails

**Symptoms**:
```
error: Microsoft Visual C++ 14.0 or greater is required
```

**Solution (Windows)**:
1. Install Visual Studio Build Tools
2. Or use pre-built wheels: `pip install faiss-cpu==1.12.0`

**Solution (Linux)**:
```bash
sudo apt install build-essential python3-dev
pip install faiss-cpu
```

**Solution (macOS)**:
```bash
brew install libomp
pip install faiss-cpu
```

### Issue 2: OpenAI Authentication Error

**Symptoms**:
```
openai.AuthenticationError: Invalid API key
```

**Solution**:
1. Verify `.env` file exists: `cat .env`
2. Check API key format: Should start with `sk-`
3. Test manually:
   ```bash
   export OPENAI_API_KEY=your-key
   python -c "import os; print(os.getenv('OPENAI_API_KEY'))"
   ```
4. Ensure no extra spaces or quotes in `.env`

### Issue 3: Streamlit Port Already in Use

**Symptoms**:
```
OSError: [Errno 48] Address already in use
```

**Solution**:
```bash
# Find process using port 8501
lsof -i :8501  # Linux/macOS
netstat -ano | findstr :8501  # Windows

# Kill the process
kill -9 <PID>  # Linux/macOS
taskkill /PID <PID> /F  # Windows

# Or use different port
streamlit run app.py --server.port 8502
```

### Issue 4: Data File Not Found

**Symptoms**:
```
Error: The data file 'data/Fetii_data.csv' was not found
```

**Solution**:
```bash
# Check if file exists
ls data/Fetii_data.csv

# Verify working directory
pwd

# Should be in project root, not subdirectory
cd /path/to/Onkar_FetiiAI
```

### Issue 5: Memory Error

**Symptoms**:
```
MemoryError: Unable to allocate array
```

**Solution**:
1. Close other applications
2. Increase swap space (Linux)
3. Use smaller embedding model:
   ```python
   VectorDB(model_name="all-MiniLM-L6-v2")  # Smaller, faster
   ```
4. Reduce batch size in PandasAI

### Issue 6: Slow Model Loading

**Symptoms**:
- First query takes 30+ seconds

**Solution** (Expected behavior):
- First run downloads Sentence-Transformers model (~400 MB)
- Subsequent runs load from cache (~5 seconds)
- To pre-download: `python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"`

---

## Advanced Setup

### Custom Data Source

To use your own CSV data:

1. **Place CSV in data directory**:
   ```bash
   cp your_data.csv data/your_data.csv
   ```

2. **Update `app.py`** (line 19):
   ```python
   data_path = 'data/your_data.csv'
   ```

3. **Restart application**

### Alternative LLM Provider

#### Using Azure OpenAI

Edit `query_engine.py`:
```python
from pandasai.llm import AzureOpenAI

class QueryEngine:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.llm = AzureOpenAI(
            api_token=os.getenv("AZURE_OPENAI_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_version="2023-05-15",
            deployment_name="your-deployment"
        )
        # ... rest of code
```

#### Using Mistral AI

```python
from pandasai.llm import Mistral

self.llm = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))
```

### Performance Tuning

#### Enable GPU Acceleration (if available)

```bash
# Install FAISS with GPU support
pip uninstall faiss-cpu
pip install faiss-gpu
```

#### Optimize for Large Datasets

Create `config.py`:
```python
ENABLE_CACHING = True
CACHE_SIZE_LIMIT = 1000  # Max cached queries
VECTOR_BATCH_SIZE = 100  # Batch embedding computation
LLM_TEMPERATURE = 0.0    # Deterministic outputs
```

Update `app.py`:
```python
import config

# In cache logic
if len(st.session_state.cache) > config.CACHE_SIZE_LIMIT:
    # Remove oldest entries
    oldest_key = list(st.session_state.cache.keys())[0]
    del st.session_state.cache[oldest_key]
```

### Development Mode

For development with auto-reload:

```bash
# Install development dependencies
pip install watchdog

# Run with debug mode
streamlit run app.py --server.runOnSave true

# Or enable logging
STREAMLIT_LOGGER_LEVEL=debug streamlit run app.py
```

### Testing Setup

```bash
# Install testing dependencies
pip install pytest pytest-cov

# Create test directory
mkdir tests

# Run tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=. --cov-report=html
```

---

## Next Steps

After successful installation:

1. **Read the documentation**: See [DOCUMENTATION.md](DOCUMENTATION.md)
2. **Explore API reference**: See [API_REFERENCE.md](API_REFERENCE.md)
3. **Try example queries**: See [README.md](README.md#usage-examples)
4. **Customize for your data**: See [Advanced Setup](#advanced-setup)

---

## Getting Help

- **GitHub Issues**: https://github.com/onkar2002406/Onkar_FetiiAI/issues
- **Streamlit Community**: https://discuss.streamlit.io/
- **PandasAI Docs**: https://docs.pandas-ai.com/

---

**Last Updated**: March 2026
**Version**: 1.0
