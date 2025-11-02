"""
Summarization module for InsightGPT
Provides map-reduce summarization and insight extraction from documents
"""
import os
from pathlib import Path
from configparser import ConfigParser
from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.chains import MapReduceDocumentsChain, ReduceDocumentsChain, StuffDocumentsChain
from langchain_core.output_parsers import StrOutputParser
from langchain_neo4j import Neo4jGraph
from src.utils.config_loader import (
    load_config as load_env_config,
    get_neo4j_config,
    get_llm_config,
)

# Global variables for lazy initialization
_graph = None
_llm = None
_config = None

def get_config():
    """Lazy load configuration"""
    global _config
    if _config is None:
        _config = load_env_config(use_env=True)
    return _config

def get_graph():
    """Lazy load Neo4j graph connection"""
    global _graph
    if _graph is None:
        try:
            neo4j_cfg = get_neo4j_config()
            os.environ["NEO4J_URI"] = neo4j_cfg["uri"]
            os.environ["NEO4J_USERNAME"] = neo4j_cfg["username"]
            os.environ["NEO4J_PASSWORD"] = neo4j_cfg["password"]
            _graph = Neo4jGraph(
                url=neo4j_cfg["uri"],
                username=neo4j_cfg["username"],
                password=neo4j_cfg["password"],
                database=os.getenv("NEO4J_DATABASE", None),
            )
        except Exception as e:
            print(f"Warning: Could not initialize Neo4j connection: {e}")
            _graph = None
    return _graph

def get_llm():
    """Lazy load LLM"""
    global _llm
    if _llm is None:
        llm_cfg = get_llm_config()
        if llm_cfg["provider"].lower() == "openai":
            from langchain_openai import ChatOpenAI
            openai_params = {
                "temperature": float(llm_cfg.get("temperature", 0.0)),
                # Lower max_tokens for faster responses
                "max_tokens": int(llm_cfg.get("max_tokens", 512)),
                # Add timeouts/retries to prevent long hangs
                "request_timeout": int(llm_cfg.get("request_timeout", 40)),
                "max_retries": int(llm_cfg.get("max_retries", 1)),
                "openai_api_key": llm_cfg.get("api_key", ""),
                "model": llm_cfg.get("model", "gpt-3.5-turbo"),
            }
            if "api_base" in llm_cfg and llm_cfg.get("api_base"):
                openai_params["openai_api_base"] = llm_cfg["api_base"]
            _llm = ChatOpenAI(**openai_params)
        elif llm_cfg["provider"].lower() == "ollama":
            from langchain_community.chat_models import ChatOllama
            options = {
                "temperature": float(llm_cfg.get("temperature", 0.0)),
                "num_ctx": int(llm_cfg.get("num_ctx", 2048)),
            }
            _llm = ChatOllama(model=llm_cfg.get("model", "hermes-2-pro-llama-3-8b"), options=options)
        else:
            raise ValueError("Invalid LLM model configuration")
    return _llm


class ResearchSummarizer:
    """Summarizes research documents using map-reduce approach"""
    
    def __init__(self, llm=None):
        self.llm = llm or get_llm()
        self.map_prompt = PromptTemplate(
            template="""You are analyzing a section of a research paper.
            
Extract the following from this text:
- Key findings and contributions
- Methodology (if present)
- Important entities (models, datasets, metrics, concepts)
- Notable results or numbers

Text section:
{text}

Provide a structured summary:""",
            input_variables=["text"]
        )
        
        self.reduce_prompt = PromptTemplate(
            template="""You are synthesizing multiple summaries of a research paper.

Combine these summaries into a comprehensive, coherent overview:
{text}

Provide a final summary with:
1. **Main Contribution**: What is the key innovation or finding?
2. **Methodology**: How was the research conducted?
3. **Key Results**: What are the main outcomes and metrics?
4. **Significance**: Why does this matter?
5. **Key Entities**: List important models, datasets, methods mentioned

Final Summary:""",
            input_variables=["text"]
        )
    
    def summarize_documents(self, documents: List[Document]) -> str:
        """
        Summarize a list of documents using map-reduce strategy
        
        Args:
            documents: List of Document objects to summarize
            
        Returns:
            str: Comprehensive summary
        """
        if not documents:
            return "No documents to summarize."
        
        # Split long documents into manageable chunks for faster map-reduce
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=150,
            separators=["\n\n", "\n", ". ", " "]
        )
        split_docs: List[Document] = []
        for doc in documents:
            split_docs.extend(text_splitter.split_documents([doc]))

        # Hard-cap total chunks to keep latency predictable
        if len(split_docs) > 12:
            split_docs = split_docs[:12]

        # Use LangChain's load_summarize_chain with map_reduce
        chain = load_summarize_chain(
            self.llm,
            chain_type="map_reduce",
            map_prompt=self.map_prompt,
            combine_prompt=self.reduce_prompt,
            verbose=False
        )
        
        summary = chain.invoke(split_docs)
        return summary.get('output_text', str(summary))
    
    def summarize_from_graph(self, paper_title: str = None, source: str = None) -> str:
        """
        Retrieve documents from Neo4j and summarize
        
        Args:
            paper_title: Title of the paper to summarize
            source: Source path/identifier
            
        Returns:
            str: Summary of the paper
        """
        # Query Neo4j - prioritize Document nodes with text content
        if paper_title:
            # Prefer Document nodes with text that match the title or source
            query = """
            MATCH (n:Document)
            WHERE n.text IS NOT NULL AND (
                toLower(coalesce(n.id, '')) CONTAINS toLower($title) OR
                toLower(coalesce(n.source, '')) CONTAINS toLower($title)
            )
            RETURN n.text as text, coalesce(n.source, 'unknown') as source
            LIMIT 5
            """
            graph = get_graph()
            if graph is None:
                return "Neo4j database not available."
            results = graph.query(query, {"title": paper_title})
        elif source:
            # Search by source
            query = """
            MATCH (n:Document)
            WHERE n.source = $source AND n.text IS NOT NULL
            RETURN n.text as text, n.source as source
            LIMIT 5
            """
            graph = get_graph()
            if graph is None:
                return "Neo4j database not available."
            results = graph.query(query, {"source": source})
        else:
            # Get all Document nodes with text first
            # Limit to 5 for faster summarization
            query = """
            MATCH (n:Document)
            WHERE n.text IS NOT NULL
            RETURN n.text as text, coalesce(n.source, 'document') as source
            LIMIT 5
            """
            graph = get_graph()
            if graph is None:
                return "Neo4j database not available."
            results = graph.query(query)
        
        if not results:
            return "No documents found in the knowledge graph."
        
        # Convert to Document objects
        documents = [
            Document(
                page_content=r['text'],
                metadata={"source": r['source']}
            ) for r in results
        ]
        
        return self.summarize_documents(documents)
    
    def extract_insights(self, text: str) -> Dict[str, Any]:
        """
        Extract structured insights from text
        
        Args:
            text: Text to analyze
            
        Returns:
            Dict containing extracted insights
        """
        insight_prompt = PromptTemplate(
            template="""Analyze this research text and extract:

1. Key Concepts: Main ideas and theories
2. Methods: Techniques and approaches used
3. Findings: Results and discoveries
4. Gaps: What's missing or needs more research
5. Applications: Practical uses

Text:
{text}

Provide insights in a structured format:""",
            input_variables=["text"]
        )
        
        chain = insight_prompt | self.llm | StrOutputParser()
        insights = chain.invoke({"text": text})
        
        return {
            "raw_insights": insights,
            "text_length": len(text),
            "source": "insight_extraction"
        }
    
    def generate_hypothesis(self, context: str, research_question: str = None) -> str:
        """
        Generate research hypotheses based on context
        
        Args:
            context: Background information and existing knowledge
            research_question: Specific research question (optional)
            
        Returns:
            str: Generated hypotheses
        """
        hypothesis_prompt = PromptTemplate(
            template="""You are a research assistant helping to generate hypotheses.

Context from existing research:
{context}

{question_text}

Based on the context, generate:
1. 3 testable hypotheses that could advance this research area
2. For each hypothesis, explain:
   - What would it test?
   - Why is it interesting?
   - How might it be tested?

Hypotheses:""",
            input_variables=["context", "question_text"]
        )
        
        question_text = f"Research Question: {research_question}" if research_question else "Generate novel hypotheses based on this context."
        
        chain = hypothesis_prompt | self.llm | StrOutputParser()
        hypotheses = chain.invoke({
            "context": context,
            "question_text": question_text
        })
        
        return hypotheses


class CitationExtractor:
    """Extract and manage citations from research papers"""
    
    def __init__(self, llm=None):
        self.llm = llm or get_llm()
        self.citation_prompt = PromptTemplate(
            template="""Extract all citations from this text. For each citation, identify:
- Authors
- Year
- Title (if available)
- Context (how it's being used)

Text:
{text}

Provide citations in this format:
[Author(s), Year] - Title - Context

Citations:""",
            input_variables=["text"]
        )
    
    def extract_citations(self, text: str) -> List[Dict[str, str]]:
        """Extract citations from text"""
        chain = self.citation_prompt | self.llm | StrOutputParser()
        citations_text = chain.invoke({"text": text})
        
        # Parse the output (simplified - could be enhanced with regex)
        citations = []
        for line in citations_text.strip().split('\n'):
            if line.strip() and not line.startswith('#'):
                citations.append({
                    "citation": line.strip(),
                    "raw_text": text[:200]  # Keep sample of original
                })
        
        return citations
    
    def build_citation_graph(self, source_paper: str, citations: List[Dict[str, str]]):
        """
        Build citation relationships in Neo4j
        
        Args:
            source_paper: The paper that contains citations
            citations: List of citation dictionaries
        """
        graph = get_graph()
        if graph is None:
            return
        
        # Create or merge the source paper node
        graph.query("""
            MERGE (p:Paper {title: $title})
            SET p.processed_date = datetime()
        """, {"title": source_paper})
        
        # Create citation relationships
        for citation in citations:
            citation_text = citation.get('citation', '')
            graph.query("""
                MERGE (p1:Paper {title: $source})
                MERGE (p2:Citation {text: $citation})
                MERGE (p1)-[r:CITES]->(p2)
                SET r.context = $context
            """, {
                "source": source_paper,
                "citation": citation_text,
                "context": citation.get('raw_text', '')
            })
    
    def get_citation_network(self, paper_title: str) -> Dict[str, Any]:
        """Get citation network for a paper"""
        graph = get_graph()
        if graph is None:
            return {"paper": paper_title, "citations": [], "citation_count": 0}
        
        query = """
        MATCH (p:Paper {title: $title})-[r:CITES]->(c:Citation)
        RETURN c.text as citation, r.context as context
        """
        results = graph.query(query, {"title": paper_title})
        
        return {
            "paper": paper_title,
            "citations": results,
            "citation_count": len(results)
        }


class LiteratureGraphBuilder:
    """Build and query literature knowledge graphs"""
    
    def __init__(self):
        self.graph = None
    
    def _get_graph(self):
        """Get graph instance"""
        if self.graph is None:
            self.graph = get_graph()
        return self.graph
    
    def build_paper_relationships(self):
        """Create relationships between papers based on shared entities"""
        graph = self._get_graph()
        if graph is None:
            return []
        
        query = """
        // Find papers that share entities
        MATCH (p1:Paper)-[:MENTIONS]->(e:__Entity__)<-[:MENTIONS]-(p2:Paper)
        WHERE id(p1) < id(p2)
        WITH p1, p2, collect(e.id) as shared_entities
        WHERE size(shared_entities) >= 2
        MERGE (p1)-[r:RELATED_TO]->(p2)
        SET r.shared_entities = shared_entities,
            r.similarity_count = size(shared_entities)
        RETURN count(r) as relationships_created
        """
        result = graph.query(query)
        return result
    
    def get_literature_map(self, topic: str = None) -> Dict[str, Any]:
        """Get literature map for visualization"""
        graph = self._get_graph()
        if graph is None:
            return {"papers": [], "topic": topic}
        
        if topic:
            query = """
            MATCH (e:__Entity__ {id: $topic})<-[:MENTIONS]-(p:Paper)
            OPTIONAL MATCH (p)-[r:RELATED_TO]-(p2:Paper)
            RETURN p.title as paper, 
                   collect(DISTINCT p2.title) as related_papers,
                   count(r) as connection_count
            LIMIT 20
            """
            results = graph.query(query, {"topic": topic})
        else:
            query = """
            MATCH (p:Paper)
            OPTIONAL MATCH (p)-[r:RELATED_TO]-(p2:Paper)
            RETURN p.title as paper,
                   collect(DISTINCT p2.title) as related_papers,
                   count(r) as connection_count
            LIMIT 20
            """
            results = graph.query(query)
        
        return {
            "papers": results,
            "topic": topic
        }


# CLI functions for testing
if __name__ == "__main__":
    import sys
    
    summarizer = ResearchSummarizer()
    citation_extractor = CitationExtractor()
    
    print("=== InsightGPT Summarization Module ===\n")
    
    if len(sys.argv) > 1:
        paper_title = sys.argv[1]
        print(f"Summarizing: {paper_title}\n")
        summary = summarizer.summarize_from_graph(paper_title=paper_title)
        print(summary)
    else:
        print("Usage: python summarizer.py 'paper_title'")
        print("\nOr use interactively:")
        print("1. Summarize from graph")
        print("2. Generate hypotheses")
        
        choice = input("\nChoice: ")
        
        if choice == "1":
            title = input("Enter paper title or source: ")
            summary = summarizer.summarize_from_graph(paper_title=title)
            print("\n" + "="*50)
            print(summary)
        elif choice == "2":
            context = input("Enter research context: ")
            question = input("Enter research question (or press Enter): ")
            hypotheses = summarizer.generate_hypothesis(context, question if question else None)
            print("\n" + "="*50)
            print(hypotheses)

