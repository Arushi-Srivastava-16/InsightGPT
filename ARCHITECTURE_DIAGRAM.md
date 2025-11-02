# InsightGPT Architecture Diagram

## System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                                InsightGPT                                        │
│                         AI-Powered Research Copilot                             │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   PDF Upload    │    │   Streamlit     │    │   Neo4j DB      │
│   Interface     │───▶│   Web App       │───▶│   (5.11+)      │
│                 │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                        │                        │
         ▼                        ▼                        ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   PDF Parser    │    │   Chat & Query  │    │   Knowledge     │
│   (Unstructured)│    │   Interface      │    │   Graph         │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                        │                        │
         ▼                        ▼                        ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   LLM Graph     │    │   LangChain     │    │   Vector Store  │
│   Transformer   │    │   Expression    │    │   (Embeddings)  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                        │                        │
         ▼                        ▼                        ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Entity &      │    │   OpenAI/Ollama │    │   Citation      │
│   Relationship  │    │   LLM           │    │   Validator     │
│   Extraction    │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Data Flow Pipeline

### 1. PDF Processing Pipeline
```
PDF Upload → Unstructured PDF Parser → Text Chunks → LLM Graph Transformer → Neo4j Graph
```

### 2. Query Processing Pipeline
```
User Query → LangChain Expression → Graph Retrieval → Vector Search → LLM Response
```

### 3. Knowledge Graph Construction
```
Text Chunks → Entity Extraction → Relationship Mapping → Graph Storage → Vector Indexing
```

## Component Interaction Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              User Interface Layer                               │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│   PDF       │  │   Chat &    │  │   Visual-   │  │   Citation  │  │   Summary   │
│   Upload    │  │   Query     │  │   izations │  │   Manager   │  │   Generator │
└─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘
       │                │                │                │                │
       ▼                ▼                ▼                ▼                ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              Processing Layer                                   │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│   PDF       │  │   Text      │  │   Entity    │  │   Graph     │  │   Vector    │
│   Parser    │  │   Chunker   │  │   Extractor │  │   Builder   │  │   Indexer   │
└─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘
       │                │                │                │                │
       ▼                ▼                ▼                ▼                ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              Data Layer                                         │
└─────────────────────────────────────────────────────────────────────────────────┘

┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│   Neo4j     │  │   Vector    │  │   Document  │  │   Entity    │  │   Relation  │
│   Database  │  │   Store     │  │   Storage   │  │   Storage   │  │   Storage   │
└─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘
```

## Knowledge Graph Schema

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              Graph Schema                                        │
└─────────────────────────────────────────────────────────────────────────────────┘

Document Nodes                    Entity Nodes                    Relationship Types
┌─────────────┐                  ┌─────────────┐                  ┌─────────────┐
│ Properties: │                  │ Properties: │                  │ Types:      │
│ - source    │                  │ - id        │                  │ - USES      │
│ - text      │                  │ - type      │                  │ - IMPLEMENTS│
│ - upload_   │                  │ - metadata  │                  │ - COMPARES  │
│   date      │                  │             │                  │ - EXTENDS   │
└─────────────┘                  └─────────────┘                  │ - BASED_ON  │
     │                                  │                        │ - IMPROVES  │
     │ MENTIONS                         │                        └─────────────┘
     ▼                                  ▼
┌─────────────┐                  ┌─────────────┐
│ Document    │◄─────────────────┤ Entity      │
│ (Paper 1)   │    MENTIONS      │ (BERT)      │
└─────────────┘                  └─────────────┘
     │                                  │
     │ MENTIONS                         │ USES
     ▼                                  ▼
┌─────────────┐                  ┌─────────────┐
│ Document    │                  │ Entity      │
│ (Paper 2)   │                  │ (Transformers)│
└─────────────┘                  └─────────────┘
```

## Technology Stack

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              Technology Stack                                   │
└─────────────────────────────────────────────────────────────────────────────────┘

Frontend                    Backend                     Database                    AI/ML
┌─────────────┐            ┌─────────────┐            ┌─────────────┐            ┌─────────────┐
│ Streamlit   │            │ Python 3.9+ │            │ Neo4j 5.11+│            │ OpenAI API  │
│ HTML/CSS    │            │ LangChain   │            │ Cypher     │            │ GPT-3.5    │
│ JavaScript  │            │ FastAPI     │            │ Vector     │            │ Embeddings │
└─────────────┘            └─────────────┘            │ Index     │            └─────────────┘
                                                      └─────────────┘

Processing                 Visualization              Configuration              Deployment
┌─────────────┐            ┌─────────────┐            ┌─────────────┐            ┌─────────────┐
│ Unstructured│            │ NetworkX    │            │ Config.ini  │            │ Docker      │
│ PDF Parser  │            │ Pyvis      │            │ .env Files  │            │ Kubernetes  │
│ JSON Repair │            │ Plotly     │            │ Environment │            │ Cloud       │
└─────────────┘            └─────────────┘            │ Variables  │            │ Platforms   │
                                                      └─────────────┘            └─────────────┘
```

## API Integration Flow

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              API Integration Flow                               │
└─────────────────────────────────────────────────────────────────────────────────┘

User Request → Streamlit → LangChain → OpenAI API → Response Processing → User Interface
     │              │           │           │              │              │
     ▼              ▼           ▼           ▼              ▼              ▼
┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐
│ Natural │  │ Web     │  │ Chain   │  │ LLM     │  │ Format  │  │ Display │
│Language │  │ Interface│  │ Orchestr│  │ Model   │  │ Response│  │ Results │
│ Query   │  │         │  │ -ation  │  │         │  │         │  │         │
└─────────┘  └─────────┘  └─────────┘  └─────────┘  └─────────┘  └─────────┘
```

## Security & Configuration

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              Security & Configuration                          │
└─────────────────────────────────────────────────────────────────────────────────┘

Authentication              Configuration              Data Protection            Monitoring
┌─────────────┐            ┌─────────────┐            ┌─────────────┐            ┌─────────────┐
│ Neo4j Auth  │            │ Environment │            │ Encryption  │            │ Logging     │
│ API Keys    │            │ Variables   │            │ at Rest     │            │ Metrics     │
│ User Roles  │            │ Config Files│            │ in Transit │            │ Alerts      │
└─────────────┘            └─────────────┘            └─────────────┘            └─────────────┘
```