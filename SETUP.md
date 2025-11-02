# 🚀 InsightGPT Setup Guide

## 📁 Project Structure

```
InsightGPT/
├── app.py                      # Main entry point (run this!)
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git ignore rules
├── README.md                   # Project overview
│
├── src/                        # Source code
│   ├── __init__.py
│   ├── core/                   # Core processing modules
│   │   ├── __init__.py
│   │   ├── summarizer.py       # Summarization & hypothesis generation
│   │   ├── citation_validator.py  # Citation management
│   │   └── llamaindex_integration.py  # LlamaIndex integration
│   │
│   ├── ui/                     # User interface
│   │   ├── __init__.py
│   │   ├── app.py              # Streamlit web interface
│   │   └── semantic_search_ui.py  # Search interface
│   │
│   └── utils/                  # Utilities
│       ├── __init__.py
│       └── config_loader.py    # Configuration loader (.env support)
│
├── config/                     # Configuration files
│   ├── env.template            # Environment variables template
│   └── config.ini.bak          # Backup config (legacy)
│
├── docs/                       # Documentation
│   ├── INSIGHTGPT_GUIDE.md
│   ├── FEATURES_SUMMARY.md
│   ├── RELEASE_NOTES.md
│   └── PROJECT_SUMMARY.txt
│
├── scripts/                    # Helper scripts
│   └── quickstart.py           # Setup wizard
│
├── tests/                      # Unit tests (to be added)
│   └── ...
│
├── graphQA.py                  # Q&A engine (legacy location)
├── pdf2graph.py                # PDF processing (legacy location)
└── res/                        # Resources (images, etc.)
```

---

## ⚙️ Environment Configuration (.env file)

### Step 1: Create Your .env File

```bash
# Windows
copy config\env.template config\.env

# Linux/Mac
cp config/env.template config/.env
```

### Step 2: Required API Keys

Open `config/.env` in a text editor and configure the following:

---

### 🔑 Required APIs & Keys

#### 1. **Neo4j Database** (REQUIRED)

Neo4j is the graph database that stores your knowledge graph.

**Setup**:
1. Download [Neo4j Desktop](https://neo4j.com/download/)
2. Create a new database
3. Set a password
4. Install APOC plugin
5. Start the database

**Add to .env**:
```env
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_actual_password_here
```

**Cost**: ✅ FREE (local installation)

---

#### 2. **OpenAI API** (Optional - for cloud AI)

Only needed if you want to use GPT-3.5/GPT-4 instead of local models.

**Setup**:
1. Go to [OpenAI API Keys](https://platform.openai.com/api-keys)
2. Create an account
3. Generate an API key
4. Add payment method (pay-as-you-go)

**Add to .env**:
```env
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-your-actual-api-key-here
OPENAI_MODEL=gpt-3.5-turbo
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
```

**Cost**: 💵 **Paid** (but cheap!)
- GPT-3.5-turbo: $0.0005/1K tokens (~$0.50 per 1M tokens)
- GPT-4: $0.03/1K tokens
- Embeddings: $0.00013/1K tokens
- Typical research paper: ~$0.10-0.50 to process
- [Pricing Details](https://openai.com/pricing)

**Alternative**: Use Ollama (see below) for FREE local models!

---

#### 3. **Ollama** (Recommended - FREE local AI)

Run AI models locally on your computer - completely free and private!

**Setup**:
1. Download [Ollama](https://ollama.com)
2. Install Ollama
3. Pull models:
```bash
ollama pull hermes-2-pro-llama-3-8b
ollama pull mxbai-embed-large
```

**Add to .env**:
```env
LLM_PROVIDER=ollama
OLLAMA_MODEL=hermes-2-pro-llama-3-8b
OLLAMA_EMBEDDING_MODEL=mxbai-embed-large
OLLAMA_NUM_CTX=2048
```

**Cost**: ✅ **100% FREE**
- No API costs
- Runs on your computer
- Complete privacy
- No internet required (after download)

**Requirements**:
- 8GB+ RAM recommended
- 10GB+ disk space for models
- Works on Mac, Windows, Linux

---

#### 4. **Zotero API** (Optional - for library integration)

Only needed if you want to import PDFs from your Zotero library.

**Setup**:
1. Go to [Zotero Settings](https://www.zotero.org/settings/keys)
2. Find your User ID
3. Create a new API key
4. Give it read permissions

**Add to .env**:
```env
ZOTERO_ENABLED=true
ZOTERO_LIBRARY_ID=your_user_id_here
ZOTERO_LIBRARY_TYPE=user
ZOTERO_API_KEY=your_zotero_api_key_here
ZOTERO_STORAGE_DIR=C:\Users\YourName\Zotero\storage\
```

**Cost**: ✅ FREE

---

### 📝 Complete .env Example

Here's a complete example for **local-only** setup (FREE):

```env
# Neo4j (Required)
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=mySecurePassword123

# LLM Provider (Ollama - Free & Local)
LLM_PROVIDER=ollama
LLM_TEMPERATURE=0.0
LLM_MAX_TOKENS=2048

# Ollama Configuration
OLLAMA_MODEL=hermes-2-pro-llama-3-8b
OLLAMA_EMBEDDING_MODEL=mxbai-embed-large
OLLAMA_NUM_CTX=2048

# Embeddings
EMBEDDINGS_PROVIDER=ollama

# Chat
CHAT_PROVIDER=ollama
CHAT_MODEL=hermes-2-pro-llama-3-8b
CHAT_TEMPERATURE=0.8

# Zotero (Optional)
ZOTERO_ENABLED=false

# PDF Processing
PDF_EXTRACT_IMAGES=false
PDF_MAX_CHAR=1000
PDF_NEW_AFTER_N_CHARS=800
PDF_COMBINE_TEXT_UNDER_N_CHARS=200
```

---

### 💰 Cost Comparison

| Setup | Neo4j | AI Models | Total Cost |
|-------|-------|-----------|------------|
| **Local (Ollama)** | FREE | FREE | ✅ $0/month |
| **Cloud (OpenAI)** | FREE | ~$5-20/month | 💵 Pay-as-you-go |
| **Hybrid** | FREE | ~$2-10/month | 💵 Minimal |

**Recommendation**: Start with Ollama (free), upgrade to OpenAI if you need faster responses or better quality.

---

## 🔧 Installation Steps

### 1. Install Python Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- LangChain & LlamaIndex
- Streamlit (web UI)
- Neo4j driver
- python-dotenv (for .env support)
- All other dependencies

### 2. Setup Neo4j

1. Install Neo4j Desktop
2. Create database with password
3. Install APOC plugin
4. Start database
5. Add password to `.env`

### 3. Setup AI Provider

**Option A: Ollama (Recommended)**
```bash
# Install from https://ollama.com
ollama pull hermes-2-pro-llama-3-8b
ollama pull mxbai-embed-large
```

**Option B: OpenAI**
- Get API key from OpenAI
- Add to `.env`

### 4. Create .env File

```bash
copy config\env.template config\.env
# Edit config\.env with your values
```

### 5. Run the Application

```bash
streamlit run app.py
```

Open: `http://localhost:8501`

---

## 🎯 Quick Start Commands

### Automated Setup
```bash
python scripts/quickstart.py
```

### Manual Steps
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Setup .env
copy config\env.template config\.env
# Edit config\.env

# 3. Start Neo4j
# Use Neo4j Desktop

# 4. Start Ollama (if using local)
ollama serve

# 5. Run app
streamlit run app.py
```

---

## 🔍 Troubleshooting

### "No .env file found"
- Make sure `.env` is in the `config/` folder
- Or place it in the root directory
- Check file name (not `.env.txt`)

### "Cannot connect to Neo4j"
- Verify Neo4j is running
- Check password in `.env`
- Check URI (default: bolt://localhost:7687)

### "OpenAI API Error"
- Verify API key is correct
- Check you have credits
- Try: `LLM_PROVIDER=ollama` instead

### "Ollama not found"
- Install from https://ollama.com
- Run `ollama serve`
- Pull models: `ollama pull hermes-2-pro-llama-3-8b`

### "Import errors"
- Run: `pip install -r requirements.txt --upgrade`
- Check Python version (3.8+ required)

---

## 📖 Configuration Options

### Choosing LLM Provider

**For Speed & Cost**: Use Ollama
```env
LLM_PROVIDER=ollama
OLLAMA_MODEL=hermes-2-pro-llama-3-8b
```

**For Quality**: Use OpenAI
```env
LLM_PROVIDER=openai
OPENAI_MODEL=gpt-4
```

**Hybrid Approach**:
```env
# Use Ollama for processing (cheap)
LLM_PROVIDER=ollama
OLLAMA_MODEL=hermes-2-pro-llama-3-8b

# Use OpenAI for chat (better quality)
CHAT_PROVIDER=openai
CHAT_MODEL=gpt-3.5-turbo
```

---

## 🔐 Security Best Practices

1. **NEVER commit .env file to Git** (already in .gitignore)
2. **Keep API keys secret**
3. **Use environment variables in production**
4. **Rotate keys regularly**
5. **Use separate keys for dev/prod**

---

## 📚 Next Steps

After setup:

1. **Upload a PDF**: Go to "Upload PDF" page
2. **Ask Questions**: Go to "Chat & Query"
3. **Generate Summary**: Go to "Summarize"
4. **Explore**: Try all features!

---

## 🆘 Need Help?

- **Documentation**: See `docs/INSIGHTGPT_GUIDE.md`
- **Issues**: GitHub Issues
- **Community**: GitHub Discussions

---

## ✅ Configuration Checklist

- [ ] Neo4j installed and running
- [ ] Neo4j password set
- [ ] `.env` file created in `config/` folder
- [ ] AI provider chosen (Ollama or OpenAI)
- [ ] If OpenAI: API key added
- [ ] If Ollama: Ollama installed and models pulled
- [ ] Python dependencies installed
- [ ] App runs successfully

---

**Ready to go!** Run `streamlit run app.py` and start researching! 🚀

