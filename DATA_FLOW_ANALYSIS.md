# InsightGPT: Complete Data Flow & Architecture Analysis

## 🏗️ System Architecture Overview

InsightGPT follows a **layered microservices architecture** with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              ARCHITECTURE LAYERS                                │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────────┐
│ 1. PRESENTATION LAYER (Streamlit UI)                                            │
│    - PDF Upload Interface                                                       │
│    - Chat & Query Interface                                                     │
│    - Visualization Dashboard                                                    │
│    - Citation Manager                                                           │
│    - Summary Generator                                                          │
└─────────────────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ 2. BUSINESS LOGIC LAYER (Core Processing)                                       │
│    - PDF Processing Pipeline                                                    │
│    - Entity Extraction Logic                                                   │
│    - Query Processing Engine                                                    │
│    - Response Generation                                                        │
└─────────────────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ 3. AI/ML LAYER (Intelligence)                                                   │
│    - OpenAI GPT-3.5-turbo                                                       │
│    - OpenAI Embeddings                                                          │
│    - LangChain Orchestration                                                    │
│    - Prompt Engineering                                                         │
└─────────────────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│ 4. DATA LAYER (Storage & Retrieval)                                            │
│    - Neo4j Graph Database                                                       │
│    - Vector Store (Embeddings)                                                  │
│    - Document Storage                                                           │
│    - Configuration Management                                                   │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## 🔄 Complete Data Flow Analysis

### Phase 1: PDF Ingestion & Processing

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              PDF PROCESSING PIPELINE                            │
└─────────────────────────────────────────────────────────────────────────────────┘

1. PDF Upload (Streamlit UI)
   ├── File validation
   ├── Metadata extraction (filename, size, upload_date)
   └── Temporary file creation

2. PDF Parsing (Unstructured Library)
   ├── Strategy Selection: "fast" (no OCR) or "hi_res" (with OCR)
   ├── Text Extraction: Raw text, tables, images
   ├── Chunking Strategy: "by_title" with configurable sizes
   └── Element Categorization: Table vs Text elements

3. Text Processing
   ├── Chunk Size: 1000-3000 characters (configurable)
   ├── Overlap: 800-2400 characters (configurable)
   ├── Combination: 200 characters minimum
   └── Metadata Preservation: Source, upload_date, processing_params

4. Document Creation
   ├── LangChain Document objects
   ├── Page content assignment
   ├── Metadata attachment
   └── Batch processing preparation
```

### Phase 2: Knowledge Extraction & Graph Construction

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         KNOWLEDGE EXTRACTION PIPELINE                          │
└─────────────────────────────────────────────────────────────────────────────────┘

1. LLM Processing (OpenAI GPT-3.5-turbo)
   ├── Prompt Engineering: Structured extraction template
   ├── Entity Recognition: Concepts, models, methods, datasets
   ├── Relationship Discovery: Connections between entities
   └── JSON Output: Structured data format

2. Entity Extraction Process
   ├── Input: Text chunk from PDF
   ├── LLM Call: Structured prompt with examples
   ├── Output: JSON with entities and relationships
   └── Validation: Filter invalid/empty entries

3. Graph Construction
   ├── Node Creation: Entities become graph nodes
   ├── Edge Creation: Relationships become graph edges
   ├── Deduplication: Prevent duplicate nodes/relationships
   ├── Type Assignment: Entity types (Model, Method, Dataset, etc.)
   └── Storage: Persist to Neo4j database

4. Vector Indexing
   ├── Embedding Generation: OpenAI embeddings API
   ├── Vector Creation: 1536-dimensional vectors
   ├── Index Building: Semantic search index
   └── Storage: Neo4j vector store
```

### Phase 3: Query Processing & Response Generation

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                            QUERY PROCESSING PIPELINE                           │
└─────────────────────────────────────────────────────────────────────────────────┘

1. Query Analysis
   ├── Input: Natural language question
   ├── Entity Extraction: Identify key concepts
   ├── Intent Classification: QA vs Hypothesis vs Summary
   └── Context Determination: Single paper vs Multi-paper

2. Retrieval Strategy (Hybrid Approach)
   ├── Graph Traversal: Follow entity relationships
   ├── Vector Similarity: Semantic search using embeddings
   ├── Full-text Search: Keyword-based entity matching
   └── Source Filtering: Limit to specific papers when requested

3. Context Assembly
   ├── Graph Results: Entity neighborhoods and relationships
   ├── Vector Results: Similar document chunks
   ├── Ranking: Relevance scoring and filtering
   └── Aggregation: Combine multiple sources

4. Response Generation
   ├── Prompt Construction: Context + Question + Instructions
   ├── LLM Processing: OpenAI GPT-3.5-turbo
   ├── Formatting: Structured, technical responses
   └── Output: Detailed answers with citations
```

## 🧠 Decision-Making Process

### 1. Configuration-Driven Decisions

```python
# Configuration Loading Priority:
1. Environment Variables (.env file)
2. Config.ini file
3. Default values

# Key Decision Points:
- LLM Provider: OpenAI vs Ollama
- Embedding Model: text-embedding-3-small vs mxbai-embed-large
- PDF Strategy: fast vs hi_res
- Chunk Sizes: 1000-3000 characters
- Temperature: 0.2 (technical accuracy)
- Max Tokens: 4096 (detailed responses)
```

### 2. Dynamic Processing Decisions

```python
# PDF Processing Decisions:
if extract_images:
    strategy = "hi_res"  # Requires Tesseract OCR
else:
    strategy = "fast"    # No OCR dependency

# Entity Extraction Decisions:
if not rel.get("head") or not rel.get("tail"):
    continue  # Skip invalid relationships

# Query Processing Decisions:
if source_filter:
    # Filter by specific paper
    query = "WHERE doc.source = $source"
else:
    # Search across all papers
    query = "MATCH (d:Document)"
```

### 3. Error Handling & Fallback Decisions

```python
# Vector Search Fallback:
try:
    vector_index = Neo4jVector.from_existing_graph(...)
except Exception:
    vector_index = None  # Continue without vector search

# Configuration Fallback:
neo4j_uri = os.getenv("NEO4J_URI", config.get("Neo4j", "uri", fallback="bolt://localhost:7687"))

# Query Fallback:
if not doc_results:
    # Try CONTAINS match
    doc_results = graph.query("WHERE d.source CONTAINS $source")
    if not doc_results:
        # Get all documents
        doc_results = graph.query("MATCH (d:Document)")
```

## 🔍 Function Decision Logic

### 1. PDF Processing Function Selection

```python
def exterat_elements_from_pdf(file_path, metadata, images=False, max_char=1000, ...):
    # Decision: Strategy selection based on image extraction
    strategy = "fast" if not images else "hi_res"
    
    # Decision: Model selection for table extraction
    model_name = "yolox"  # Best for table extraction
    
    # Decision: Chunking parameters
    raw_pdf_elements = partition_pdf(
        filename=file_path,
        extract_images_in_pdf=images,
        infer_table_structure=True,
        chunking_strategy="by_title",
        max_characters=max_char,
        new_after_n_chars=new_after_n_chars,
        combine_text_under_n_chars=combine,
        strategy=strategy,
        model_name=model_name
    )
```

### 2. Entity Extraction Function Logic

```python
def process_response(document, i, j, metadata) -> GraphDocument:
    # Decision: LLM processing
    raw_schema = chain.invoke({"input": document.page_content})
    parsed_json = json_repair.loads(raw_schema.content)
    
    # Decision: Validation and filtering
    for rel in parsed_json:
        if not rel.get("head") or not rel.get("tail") or not rel.get("relation"):
            continue  # Skip invalid relationships
        
        # Decision: Type assignment
        rel["head_type"] = rel["head_type"] if rel["head_type"] else "Unknown"
        rel["tail_type"] = rel["tail_type"] if rel["tail_type"] else "Unknown"
        
        # Decision: Node creation
        nodes_set.add((head_id, rel["head_type"]))
        nodes_set.add((tail_id, rel["tail_type"]))
```

### 3. Query Processing Function Logic

```python
def create_context(input_data):
    question = input_data.get("question", "")
    source_filter = input_data.get("source_filter")
    
    # Decision: Source filtering
    if source_filter:
        # Try exact match first
        doc_results = graph.query("WHERE d.source = $source", {"source": source_filter})
        
        if not doc_results:
            # Fallback: CONTAINS match
            doc_results = graph.query("WHERE d.source CONTAINS $source", {"source": source_filter})
            
            if not doc_results:
                # Final fallback: All documents
                doc_results = graph.query("MATCH (d:Document)")
    
    # Decision: Vector search availability
    if vector_index is not None:
        similar_docs = vector_index.similarity_search_with_score(question, k=12)
        # Filter by source if provided
    else:
        # Graph-only retrieval
        unstructured_data = []
```

## 📊 Performance Optimization Decisions

### 1. Chunk Size Optimization
```python
# Decision: Larger chunks reduce LLM calls
max_char = 3000  # Increased from 1000
new_after_n_chars = 2400  # Increased from 800

# Result: Fewer chunks = Faster processing
```

### 2. Caching Strategy
```python
# Decision: Pickle files for processed documents
if os.path.exists("output.pkl"):
    with open("output.pkl", "rb") as f:
        text_summaries = pickle.load(f)
else:
    # Process documents
    text_summaries = [process_response(document, i, len(documents), metadata) for i, document in enumerate(documents)]
    with open("output.pkl", "wb") as f:
        pickle.dump(text_summaries, f)
```

### 3. Query Optimization
```python
# Decision: Limit results for performance
LIMIT 5  # For summarization
LIMIT 20  # For document retrieval
LIMIT 12  # For vector search
```

## 🎯 Key Architectural Decisions

### 1. **Hybrid Search Strategy**
- Combines graph traversal + vector similarity
- Provides both structured and semantic search
- Fallback mechanisms for missing components

### 2. **Configuration-Driven Architecture**
- Environment variables override config files
- Default values ensure system stability
- Runtime configuration changes

### 3. **Error-Resilient Design**
- Graceful degradation (vector search optional)
- Multiple fallback strategies
- Comprehensive error handling

### 4. **Modular Component Design**
- Clear separation of concerns
- Independent processing pipelines
- Reusable components

This architecture enables InsightGPT to intelligently process research papers, extract knowledge, and provide accurate, contextual responses through a sophisticated decision-making process that adapts to different scenarios and requirements.
