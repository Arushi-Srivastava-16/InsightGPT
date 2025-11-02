# Streamlit Cloud Deployment Guide

## Quick Fix for Current Error

The error you're seeing is because the Neo4j connection is trying to initialize before the environment variables are set. This has been fixed in the latest code.

## Setting up Streamlit Cloud

### 1. Deploy to Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Connect your GitHub account
3. Select your repository: `Arushi-Srivastava-16/InsightGPT`
4. Set the main file path: `src/ui/app_single_page.py`
5. Click "Deploy"

### 2. Configure Secrets

In your Streamlit Cloud app settings, add these secrets:

```toml
# Neo4j Configuration (Required)
NEO4J_URI = "neo4j://your-neo4j-host:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "your-neo4j-password"

# OpenAI Configuration (Required)
OPENAI_API_KEY = "your-openai-api-key"
OPENAI_MODEL = "gpt-3.5-turbo"

# Optional: LLM Configuration
LLM_PROVIDER = "OpenAI"
LLM_TEMPERATURE = 0.0
LLM_MAX_TOKENS = 2048

# Optional: Embedding Configuration
EMBEDDINGS_PROVIDER = "OpenAI"
EMBEDDINGS_MODEL = "text-embedding-3-small"

# Optional: Chat Configuration
CHAT_PROVIDER = "OpenAI"
CHAT_MODEL = "gpt-3.5-turbo"
CHAT_TEMPERATURE = 0.8
CHAT_MAX_TOKENS = 2048
```

### 3. Neo4j Setup Options

#### Option A: Neo4j AuraDB (Recommended for Cloud)
1. Go to [neo4j.com/cloud/aura](https://neo4j.com/cloud/aura)
2. Create a free AuraDB instance
3. Use the connection URI, username, and password in your secrets

#### Option B: Local Neo4j (Development Only)
- Not recommended for Streamlit Cloud deployment
- Use ngrok or similar to expose local Neo4j (not secure for production)

### 4. Environment Variables Priority

The app will check for configuration in this order:
1. Streamlit secrets (recommended for cloud deployment)
2. Environment variables
3. config.ini file (fallback)

## Troubleshooting

### Common Issues

1. **Neo4j Connection Error**: Make sure your Neo4j instance is accessible from the internet and credentials are correct
2. **OpenAI API Error**: Verify your API key is valid and has sufficient credits
3. **Import Errors**: Check that all dependencies are in requirements.txt

### Testing Locally

Before deploying, test locally with environment variables:

```bash
# Set environment variables
export NEO4J_URI="your-neo4j-uri"
export NEO4J_USERNAME="neo4j"
export NEO4J_PASSWORD="your-password"
export OPENAI_API_KEY="your-api-key"

# Run Streamlit
streamlit run src/ui/app_single_page.py
```

## Security Notes

- Never commit real API keys or passwords to Git
- Use Streamlit secrets for sensitive configuration
- The config.ini file contains placeholder values only
- Real credentials should be set via environment variables or Streamlit secrets

## What Was Fixed

### Fix 1: Neo4j Connection Error
The original error was caused by:
1. Neo4j connection being initialized at import time
2. Environment variables not being set before import

The fix:
1. Implemented lazy loading for Neo4j and LLM connections
2. Added Streamlit secrets support to config loader
3. Deferred initialization until actually needed
4. Added proper error handling for missing connections

### Fix 2: Dependency Installation Error
Additional issues fixed:
1. **Removed problematic system packages**: Streamlined `packages.txt` to only essential libraries
2. **Made unstructured optional**: The `unstructured[pdf]` library can cause install issues, so it's now optional with graceful fallback
3. **Fixed import errors**: Updated imports to use `langchain_core.documents` instead of deprecated `langchain.schema`
4. **Improved PDF fallback**: Enhanced the fallback mechanism to work even when unstructured is not available

The app now uses simpler PDF processing libraries (PyMuPDF, pypdf, pdfminer.six) that work reliably on Streamlit Cloud.
