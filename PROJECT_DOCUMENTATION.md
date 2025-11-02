# InsightGPT - Complete Project Documentation

## 🎯 Project Overview
InsightGPT is an AI-powered research copilot that transforms academic papers into interactive knowledge graphs, enabling intelligent querying, hypothesis generation, and citation management.

## 🏗️ Architecture Components

### Frontend Layer
- **Streamlit Web Interface**: Single-page application with multiple sections
- **PDF Upload**: Drag-and-drop interface for research papers
- **Chat Interface**: Real-time Q&A with research papers
- **Visualizations**: Interactive knowledge graphs and analytics
- **Citation Management**: Validation and auto-citation features

### Processing Layer
- **PDF Parser**: Uses Unstructured library with "fast" strategy
- **Text Chunking**: Configurable chunk sizes (1000-3000 chars)
- **LLM Integration**: OpenAI GPT-3.5-turbo or Ollama models
- **Graph Transformer**: Extracts entities and relationships

### Data Layer
- **Neo4j Database**: Graph database (requires 5.11+ for vector search)
- **Vector Store**: Embeddings for semantic search
- **Document Storage**: Text chunks with metadata
- **Knowledge Graph**: Entities, relationships, and connections

## 🔄 Data Processing Pipeline

### Phase 1: PDF Ingestion
1. **Upload**: User uploads PDF via Streamlit interface
2. **Parsing**: Unstructured library extracts text, tables, images
3. **Chunking**: Text split into manageable segments
4. **Metadata**: Source, upload date, processing parameters

### Phase 2: Knowledge Extraction
1. **LLM Processing**: Each chunk sent to LLM for entity extraction
2. **Entity Recognition**: Identifies concepts, models, methods, datasets
3. **Relationship Mapping**: Discovers connections between entities
4. **Validation**: Filters out invalid or empty relationships

### Phase 3: Graph Construction
1. **Node Creation**: Entities become graph nodes with types
2. **Edge Creation**: Relationships become graph edges
3. **Deduplication**: Prevents duplicate nodes and relationships
4. **Storage**: Persists to Neo4j database

### Phase 4: Vector Indexing
1. **Embedding Generation**: Creates vector representations
2. **Index Creation**: Builds semantic search index
3. **Hybrid Search**: Combines graph and vector search

## 🔍 Query Processing Pipeline

### Input Processing
1. **Question Analysis**: Extracts key entities from user query
2. **Context Retrieval**: Searches knowledge graph for relevant information
3. **Source Filtering**: Limits results to specific papers when requested

### Retrieval Methods
1. **Graph Traversal**: Follows entity relationships
2. **Vector Similarity**: Semantic search using embeddings
3. **Full-text Search**: Keyword-based entity matching
4. **Hybrid Approach**: Combines multiple retrieval methods

### Response Generation
1. **Context Assembly**: Combines retrieved information
2. **LLM Processing**: Generates detailed, technical responses
3. **Citation Integration**: Includes source references
4. **Formatting**: Structures output for readability

## 🛠️ Technical Stack

### Core Technologies
- **Python 3.9+**: Main programming language
- **Streamlit**: Web application framework
- **Neo4j**: Graph database (5.11+ required)
- **LangChain**: LLM orchestration framework
- **OpenAI API**: Language model provider
- **Unstructured**: PDF processing library

### Key Libraries
- **langchain-neo4j**: Neo4j integration
- **langchain-openai**: OpenAI integration
- **langchain-experimental**: Graph transformers
- **networkx**: Graph analysis
- **pyvis**: Interactive visualizations
- **plotly**: Data visualization
- **pydantic**: Data validation

## 📊 Data Models

### Graph Schema
```
Document
├── Properties: source, text, upload_date
├── Relationships: MENTIONS → Entity

Entity (__Entity__)
├── Properties: id, type
├── Relationships: Various relationship types

Node Types:
- Document: Text chunks from papers
- Entity: Extracted concepts, models, methods
- Relationship: Connections between entities
```

### Configuration Schema
```
Neo4j: URI, username, password
LLM: Provider (OpenAI/Ollama), model, temperature, max_tokens
Embeddings: Provider, model
PDF: Extract images, chunk sizes
Chat: Model, temperature, max_tokens
```

## 🚀 Key Features

### 1. Intelligent PDF Processing
- Extracts text, tables, and images
- Configurable chunking strategies
- Progress tracking and error handling
- Support for various PDF formats

### 2. Knowledge Graph Construction
- Automatic entity extraction
- Relationship discovery
- Graph validation and cleaning
- Incremental updates

### 3. Advanced Querying
- Natural language questions
- Context-aware responses
- Source-specific filtering
- Multi-paper analysis

### 4. Hypothesis Generation
- Research question suggestions
- Testable predictions
- Methodology recommendations
- Gap identification

### 5. Citation Management
- Format validation (APA, MLA, Chicago, Harvard)
- Knowledge graph verification
- Auto-citation suggestions
- Reference checking

### 6. Interactive Visualizations
- Network graphs with pyvis
- Statistical charts with plotly
- Entity co-occurrence heatmaps
- Relationship distribution analysis

## 🔧 Configuration Options

### Environment Variables
```bash
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password
OPENAI_API_KEY=your_api_key
LLM_PROVIDER=openai
EMBEDDINGS_PROVIDER=openai
```

### Config File (config.ini)
- Database connection settings
- LLM model configurations
- PDF processing parameters
- Embedding model settings

## 📈 Performance Considerations

### Optimization Strategies
- **Chunk Size**: Larger chunks (3000 chars) reduce LLM calls
- **Batch Processing**: Parallel entity extraction
- **Caching**: Pickle files for processed documents
- **Vector Indexing**: Semantic search acceleration

### Scalability
- **Horizontal**: Multiple Neo4j instances
- **Vertical**: Increased memory and CPU
- **Caching**: Redis for frequent queries
- **CDN**: Static asset delivery

## 🔒 Security & Privacy

### Data Protection
- Local processing option (Ollama)
- API key management
- Secure database connections
- No data transmission to third parties (with local LLM)

### Access Control
- Database authentication
- API key rotation
- Environment variable protection
- Secure configuration management

## 🚀 Deployment Options

### Local Development
```bash
pip install -r requirements.txt
streamlit run app.py
```

### Production Deployment
- Docker containerization
- Cloud platforms (AWS, GCP, Azure)
- Kubernetes orchestration
- Load balancing and scaling

## 📋 Use Cases

### Academic Research
- Literature review automation
- Research gap identification
- Citation network analysis
- Hypothesis generation

### Corporate R&D
- Technical document analysis
- Knowledge management
- Competitive intelligence
- Patent analysis

### Education
- Interactive learning tools
- Research methodology training
- Citation practice
- Knowledge visualization

## 🔮 Future Enhancements

### Planned Features
- Multi-language support
- Advanced visualization options
- Collaborative workspaces
- API endpoints for integration
- Mobile application
- Real-time collaboration

### Technical Improvements
- Performance optimization
- Enhanced error handling
- Advanced caching strategies
- Machine learning integration
- Automated testing suite







