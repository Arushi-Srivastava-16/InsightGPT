# 🔬 InsightGPT - AI-Powered Research Copilot

## Complete Feature Guide & Documentation

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Core Features](#core-features)
4. [Installation & Setup](#installation--setup)
5. [Usage Guide](#usage-guide)
6. [API Reference](#api-reference)
7. [Advanced Features](#advanced-features)
8. [Troubleshooting](#troubleshooting)

---

## Overview

**InsightGPT** is a comprehensive AI-powered research assistant that combines:
- **Graph RAG (Retrieval Augmented Generation)** for intelligent document understanding
- **Knowledge Graph Construction** using Neo4j
- **Multi-modal AI** with support for both OpenAI and local models (Ollama)
- **Interactive Web Interface** built with Streamlit
- **Advanced Citation Management** with validation
- **Hypothesis Generation** for research ideation
- **Literature Network Analysis** and visualization

### Why InsightGPT?

Traditional RAG systems use simple vector similarity. InsightGPT goes beyond by:
1. **Understanding Relationships**: Entities are connected in a knowledge graph
2. **Hybrid Retrieval**: Combines semantic search + graph traversal
3. **Citation Awareness**: Tracks paper citations and validates references
4. **Research Workflows**: Not just Q&A - supports summarization, hypothesis generation, and literature mapping

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    InsightGPT Architecture                   │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌─────────────┐        ┌──────────────┐                    │
│  │  Streamlit  │───────▶│   FastAPI    │                    │
│  │  Frontend   │        │   (Future)   │                    │
│  └─────────────┘        └──────────────┘                    │
│         │                       │                            │
│         ▼                       ▼                            │
│  ┌──────────────────────────────────────┐                   │
│  │         Core Processing Layer         │                   │
│  ├──────────────────────────────────────┤                   │
│  │  • pdf2graph.py    - PDF Processing  │                   │
│  │  • graphQA.py      - Q&A Engine      │                   │
│  │  • summarizer.py   - Summarization   │                   │
│  │  • citation_*      - Citation Mgmt   │                   │
│  └──────────────────────────────────────┘                   │
│         │                       │                            │
│         ▼                       ▼                            │
│  ┌─────────────┐        ┌──────────────┐                    │
│  │  LangChain  │        │  LlamaIndex  │                    │
│  │  (Primary)  │        │  (Enhanced)  │                    │
│  └─────────────┘        └──────────────┘                    │
│         │                       │                            │
│         └───────────┬───────────┘                            │
│                     ▼                                        │
│  ┌──────────────────────────────────────┐                   │
│  │           Storage Layer               │                   │
│  ├──────────────────────────────────────┤                   │
│  │  • Neo4j (Graph DB)                  │                   │
│  │  • Vector Index (Embeddings)         │                   │
│  │  • Citation Network                  │                   │
│  └──────────────────────────────────────┘                   │
│                     │                                        │
│                     ▼                                        │
│  ┌──────────────────────────────────────┐                   │
│  │            LLM Layer                  │                   │
│  ├──────────────────────────────────────┤                   │
│  │  • OpenAI API (GPT-3.5/4)            │                   │
│  │  • Ollama (Local Models)             │                   │
│  │  • llama.cpp (Alternative)           │                   │
│  └──────────────────────────────────────┘                   │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

---

## Core Features

### 1. 📄 Smart PDF Processing

**Module**: `pdf2graph.py`

Converts PDFs into structured knowledge graphs.

**Capabilities**:
- Extract text, tables, and metadata
- Identify entities (models, datasets, methods, concepts)
- Extract relationships between entities
- Build Neo4j knowledge graph
- Support for Zotero integration

**Example**:
```python
from pdf2graph import process_document

# Process a PDF
metadata = {"source": "paper.pdf", "author": "Smith et al."}
process_document(
    file_path="paper.pdf",
    meta=metadata,
    images=False,
    max_char=1000
)
```

**How It Works**:
1. PDF → Text extraction (unstructured library)
2. Text → Entity/Relationship extraction (LLM)
3. Entities/Relationships → Neo4j graph
4. Text → Embeddings → Vector store

---

### 2. 💬 Intelligent Q&A System

**Module**: `graphQA.py`

Answers questions using hybrid retrieval (graph + vector).

**Capabilities**:
- Extract entities from questions
- Graph traversal for context
- Vector similarity search
- Combine structured + unstructured data
- Conversational memory

**Example**:
```python
from graphQA import chain

# Ask a question
response = chain.invoke({"question": "What is the main contribution?"})
print(response)
```

**Query Flow**:
```
Question → Entity Extraction → Graph Retrieval
                              ↓
                         Vector Search
                              ↓
                    Combine Contexts → LLM → Answer
```

---

### 3. 📚 Map-Reduce Summarization

**Module**: `summarizer.py`

Generates comprehensive summaries using map-reduce strategy.

**Capabilities**:
- Map: Summarize individual sections
- Reduce: Combine into coherent overview
- Extract key findings, methods, results
- Generate insights and hypotheses

**Example**:
```python
from summarizer import ResearchSummarizer

summarizer = ResearchSummarizer()

# Summarize from knowledge graph
summary = summarizer.summarize_from_graph(paper_title="transformer")
print(summary)

# Generate hypotheses
hypotheses = summarizer.generate_hypothesis(
    context="Recent advances in attention mechanisms...",
    research_question="How can we improve efficiency?"
)
print(hypotheses)
```

**Features**:
- ✅ Map-reduce chain for long documents
- ✅ Structured output (contribution, methods, results, significance)
- ✅ Hypothesis generation based on context
- ✅ Insight extraction

---

### 4. 📖 Citation Management & Validation

**Modules**: `citation_validator.py`, `summarizer.py` (CitationExtractor)

Comprehensive citation handling.

**Capabilities**:
- Extract citations from text
- Validate citation formats (APA, MLA, Chicago, etc.)
- Check citations against knowledge graph
- Validate citation context (appropriate usage)
- Auto-generate citations from metadata
- Build citation network in Neo4j

**Example**:
```python
from citation_validator import CitationValidator, AutoCiter

validator = CitationValidator()

# Validate format
result = validator.validate_citation_format(
    "(Smith, 2023)",
    expected_style="apa"
)

# Batch validate
report = validator.batch_validate_citations(
    text="According to Smith (2023), the model achieves...",
    style="apa"
)
print(f"Accuracy: {report['accuracy']}%")

# Auto-cite
auto_citer = AutoCiter()
citation = auto_citer.auto_generate_intext_citation(
    claim="The model achieves 95% accuracy",
    paper_metadata={"authors": "Smith, J.", "year": "2023"},
    style="apa"
)
```

**Citation Graph**:
```cypher
(Paper)-[:CITES]->(Citation)
(Paper)-[:RELATED_TO]->(Paper)
```

---

### 5. 🔍 Semantic Search & Exploration

**Modules**: `semantic_search_ui.py`, `llamaindex_integration.py`

Advanced search with visualization.

**Capabilities**:
- Semantic similarity search
- Graph-based search
- Hybrid search (combines both)
- Network visualization
- Interactive exploration
- Filter by metadata

**Search Methods**:

1. **Semantic Search** (LlamaIndex):
   - Vector similarity
   - Fast retrieval
   - Works with embeddings

2. **Graph Search** (Neo4j):
   - Entity relationships
   - Path traversal
   - Context-aware

3. **Hybrid**:
   - Best of both worlds
   - Higher accuracy

**Example**:
```python
from llamaindex_integration import LlamaIndexManager, SemanticSearchEngine

manager = LlamaIndexManager()
search_engine = SemanticSearchEngine(manager)

# Semantic search
results = search_engine.semantic_search(
    query="attention mechanism",
    filters={"year": "2023"},
    top_k=10
)

# Multi-query
results = search_engine.multi_query_search(
    queries=["attention", "transformer", "efficiency"],
    aggregation="intersection"
)
```

**Visualizations**:
- 📊 Score distribution histogram
- 🥧 Entity type pie chart
- 🕸️ Network graph (query + results + connections)
- 📋 Results table with download

---

### 6. 🕸️ Literature Network Analysis

**Module**: `summarizer.py` (LiteratureGraphBuilder)

Build and explore literature networks.

**Capabilities**:
- Identify papers sharing entities
- Build paper-to-paper relationships
- Generate literature maps
- Topic-focused exploration

**Example**:
```python
from summarizer import LiteratureGraphBuilder

builder = LiteratureGraphBuilder()

# Build relationships
builder.build_paper_relationships()

# Get literature map
lit_map = builder.get_literature_map(topic="transformer")
for paper in lit_map['papers']:
    print(f"{paper['paper']}: {paper['connection_count']} connections")
```

**Graph Schema**:
```cypher
(Paper)-[:MENTIONS]->(Entity)
(Paper)-[:RELATED_TO {shared_entities: []}]->(Paper)
(Paper)-[:CITES]->(Citation)
```

---

## Installation & Setup

### Prerequisites

- **Python 3.8+**
- **Neo4j Desktop** (or Neo4j server)
- **Ollama** (optional, for local models)
- **OpenAI API key** (optional, for GPT models)

### Step 1: Clone & Install

```bash
# Clone repository
git clone https://github.com/your-username/InsightGPT.git
cd InsightGPT

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Neo4j Setup

1. **Download** [Neo4j Desktop](https://neo4j.com/download/)

2. **Create Database**:
   - New Project → Add → Local DBMS
   - Set password (remember this!)
   - Install APOC plugin

3. **Start Database**

### Step 3: Configuration

Copy and edit config:

```bash
cp config.ini.bak config.ini
```

Edit `config.ini`:

```ini
[Neo4j]
uri = bolt://localhost:7687
username = neo4j
password = YOUR_PASSWORD_HERE

[LLM]
llm = Ollama  # or OpenAI
temperature = 0.0
max_tokens = 2048

[Ollama]
model = interstellarninja/hermes-2-pro-llama-3-8b
num_ctx = 2048

[OpenAI]
model = gpt-3.5-turbo
api_key = sk-YOUR_KEY_HERE

[Embeddings]
embeddings = Ollama  # or OpenAI
model = mxbai-embed-large

[Zotero]
enabled = True
library_id = YOUR_ID
api_key = YOUR_KEY
Zotero_dir = /path/to/zotero/storage/

[PDF]
extract_images = False
max_char = 1000
new_after_n_chars = 800
combine_text_under_n_chars = 200
```

### Step 4: (Optional) Ollama Setup

```bash
# Install Ollama
# Visit: https://ollama.com

# Pull models
ollama pull hermes-2-pro-llama-3-8b
ollama pull mxbai-embed-large
```

---

## Usage Guide

### 🌐 Web Interface (Recommended)

```bash
streamlit run app.py
```

Open browser to: `http://localhost:8501`

**Features**:
- 📄 Upload PDFs
- 💬 Chat interface
- 🔍 Semantic search
- 📚 Summarization
- 🧠 Hypothesis generation
- 📖 Citation management
- 🕸️ Literature graphs

### 🖥️ Command Line

#### Process PDFs

```bash
python pdf2graph.py
```

Follow prompts to:
- Enter PDF path
- Search Zotero library
- Process all Zotero items

#### Chat with Documents

```bash
python graphQA.py
```

Enter questions:
```
>>> What is the main contribution of the paper?
>>> How does the model compare to baselines?
>>> exit
```

#### Generate Summaries

```bash
python summarizer.py "paper_title"
```

Or interactive:
```bash
python summarizer.py
```

#### Validate Citations

```bash
python citation_validator.py
```

Choose options:
1. Validate format
2. Validate context
3. Check in graph
4. Batch validate
5. Suggest citations

---

## API Reference

### ResearchSummarizer

```python
class ResearchSummarizer:
    def summarize_documents(self, documents: List[Document]) -> str:
        """Summarize list of documents using map-reduce"""
    
    def summarize_from_graph(self, paper_title: str = None) -> str:
        """Retrieve and summarize from Neo4j"""
    
    def extract_insights(self, text: str) -> Dict:
        """Extract structured insights"""
    
    def generate_hypothesis(self, context: str, research_question: str = None) -> str:
        """Generate research hypotheses"""
```

### CitationValidator

```python
class CitationValidator:
    def validate_citation_format(self, citation: str, expected_style: str) -> Dict:
        """Validate citation format"""
    
    def validate_citation_context(self, citation: str, context: str) -> Dict:
        """Validate appropriate usage"""
    
    def check_citation_exists(self, citation: str) -> Dict:
        """Check if citation exists in graph"""
    
    def batch_validate_citations(self, text: str, style: str) -> Dict:
        """Validate all citations in text"""
```

### LlamaIndexManager

```python
class LlamaIndexManager:
    def create_index_from_neo4j(self, label: str = "Document") -> VectorStoreIndex:
        """Create index from Neo4j data"""
    
    def query(self, question: str, similarity_top_k: int = 5) -> str:
        """Query using LlamaIndex"""
    
    def retrieve_relevant_docs(self, query: str, top_k: int = 5) -> List[Dict]:
        """Retrieve without generating answer"""
    
    def hybrid_query(self, question: str) -> Dict:
        """Query using both LlamaIndex and LangChain"""
```

### SemanticSearchEngine

```python
class SemanticSearchEngine:
    def semantic_search(self, query: str, filters: Dict = None, top_k: int = 10) -> List[Dict]:
        """Semantic search with filters"""
    
    def multi_query_search(self, queries: List[str], aggregation: str = 'union') -> List[Dict]:
        """Multi-query aggregated search"""
```

---

## Advanced Features

### 1. Custom Entity Extraction

Modify `pdf2graph.py` prompt to extract domain-specific entities:

```python
examples = [
    {
        "text": "The protein binding affinity was measured at 50nM",
        "head": "protein",
        "head_type": "Molecule",
        "relation": "HAS_AFFINITY",
        "tail": "50nM",
        "tail_type": "Measurement"
    }
]
```

### 2. Custom Summarization Templates

Modify `summarizer.py` prompts:

```python
self.reduce_prompt = PromptTemplate(
    template="""Synthesize for a biology audience:
{text}

Summary should include:
1. Biological significance
2. Molecular mechanisms
3. Experimental evidence
4. Clinical implications
""",
    input_variables=["text"]
)
```

### 3. Neo4j Cypher Queries

Direct database queries:

```python
from langchain_community.graphs import Neo4jGraph

graph = Neo4jGraph()

# Find most connected entities
query = """
MATCH (e:__Entity__)
WITH e, size((e)--()) as degree
ORDER BY degree DESC
LIMIT 10
RETURN e.id as entity, degree
"""
results = graph.query(query)
```

### 4. Batch Processing

Process multiple PDFs:

```python
import os
from pdf2graph import process_document

pdf_dir = "papers/"
for filename in os.listdir(pdf_dir):
    if filename.endswith(".pdf"):
        process_document(
            os.path.join(pdf_dir, filename),
            {"source": filename}
        )
```

---

## Troubleshooting

### Neo4j Connection Issues

**Error**: `Failed to establish connection`

**Solution**:
1. Check Neo4j is running
2. Verify `config.ini` credentials
3. Test connection:
```bash
python -c "from langchain_community.graphs import Neo4jGraph; graph = Neo4jGraph(); print('OK')"
```

### Ollama Model Issues

**Error**: `Model not found`

**Solution**:
```bash
ollama list  # Check installed models
ollama pull model-name  # Install missing model
```

### Memory Issues

**Error**: `Out of memory`

**Solution**:
- Reduce `num_ctx` in config
- Process smaller PDFs
- Reduce `max_char` in PDF settings

### LlamaIndex Import Errors

**Error**: `No module named 'llama_index'`

**Solution**:
```bash
pip install llama-index llama-index-vector-stores-neo4jvector
```

### Streamlit Port Issues

**Error**: `Port 8501 already in use`

**Solution**:
```bash
streamlit run app.py --server.port 8502
```

---

## Performance Tips

1. **Use Local Models**: Ollama for faster response (no API calls)
2. **Optimize Chunking**: Adjust `max_char` based on document type
3. **Index Management**: Rebuild Neo4j indexes periodically
4. **Batch Processing**: Process multiple PDFs in one session
5. **Cache Results**: LlamaIndex caches embeddings

---

## Roadmap

- [ ] Multi-language support
- [ ] PDF annotation export
- [ ] Collaborative features
- [ ] REST API
- [ ] Docker deployment
- [ ] Cloud deployment guides
- [ ] More visualization types
- [ ] Export to Obsidian/Notion
- [ ] Voice interface
- [ ] Mobile app

---

## Contributing

Contributions welcome! Areas to explore:
- New visualization types
- Additional LLM integrations
- Domain-specific templates
- Performance optimizations
- Bug fixes

---

## License

See [LICENSE](LICENSE) file.

---

## Support

- **Issues**: GitHub Issues
- **Discussions**: GitHub Discussions
- **Email**: support@insightgpt.ai

---

## Citation

If you use InsightGPT in your research:

```bibtex
@software{insightgpt2024,
  title={InsightGPT: AI-Powered Research Copilot},
  author={Your Name},
  year={2024},
  url={https://github.com/your-username/InsightGPT}
}
```

---

**Built with ❤️ for researchers, by researchers**

🔬 **InsightGPT** - Making research accessible and insightful

