import os
import re
from pathlib import Path
from configparser import ConfigParser
from langchain_community.vectorstores import Neo4jVector
from langchain_neo4j import Neo4jGraph
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import Tuple, List, Optional
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_community.vectorstores.neo4j_vector import remove_lucene_chars
import ast
from langchain_core.runnables import (
    RunnableBranch,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser

from src.utils.config_loader import (
    load_config as load_env_config,
    get_neo4j_config,
    get_llm_config,
)

# Load configuration
config = load_env_config(use_env=True)

# Initialize graph and LLM
neo4j_cfg = get_neo4j_config()
os.environ["NEO4J_URI"] = neo4j_cfg["uri"]
os.environ["NEO4J_USERNAME"] = neo4j_cfg["username"]
os.environ["NEO4J_PASSWORD"] = neo4j_cfg["password"]
graph = Neo4jGraph(
    url=neo4j_cfg["uri"],
    username=neo4j_cfg["username"],
    password=neo4j_cfg["password"],
    database=os.getenv("NEO4J_DATABASE", None),
)

llm_provider = get_llm_config()["provider"].capitalize()
if llm_provider == "Openai":
    from langchain_openai import ChatOpenAI
    openai_params = {
        "temperature": float(config.get("LLM", "temperature", fallback="0.0")),
        "max_tokens": int(config.get("LLM", "max_tokens", fallback="2048")),
        "openai_api_key": config.get("OpenAI", "api_key", fallback=""),
        "openai_api_base": config.get("OpenAI", "api_base", fallback=None),
        "stop": config.get("LLM", "stop", fallback=None),
    }
    llm = ChatOpenAI(**openai_params)
elif llm_provider == "Ollama":
    from langchain_community.chat_models import ChatOllama
    options = {
        "temperature": float(config.get("LLM", "temperature", fallback="0.0")),
        "num_ctx": int(config.get("Ollama", "num_ctx", fallback="2048")),
        "stop": config.get("LLM", "stop", fallback=None),
    }
    llm = ChatOllama(model=config.get("Ollama", "model", fallback="hermes-2-pro-llama-3-8b"), options=options)
else:
    raise ValueError("Invalid LLM model configuration")

if config["Embeddings"]["model"] == '':
    embeddings = llm
elif config["Embeddings"]["embeddings"] == "Ollama":
    from langchain_community.embeddings import OllamaEmbeddings
    embeddings = OllamaEmbeddings(model=config["Embeddings"]["model"])
elif config["Embeddings"]["embeddings"].lower() == "openai":
    from langchain_openai import OpenAIEmbeddings
    embeddings = OpenAIEmbeddings(model=config["Embeddings"]["model"])

# Try to initialize vector index; if Neo4j version < 5.11, fall back to graph-only
vector_index = None
try:
    vector_index = Neo4jVector.from_existing_graph(
        embeddings,
        search_type="hybrid",
        node_label="Document",
        text_node_properties=["text"],
        embedding_node_property="embedding",
    )
except Exception as e:
    # Fallback: proceed without vector search (e.g., Neo4j < 5.11 has no native vector index)
    # Silently continue without vector index
    vector_index = None

# Retriever

try:
    graph.query(
        "CREATE FULLTEXT INDEX entity IF NOT EXISTS FOR (e:__Entity__) ON EACH [e.id]")
except Exception as e:
    # Index creation can fail if database is not available or index already exists
    print(f"Warning: Could not create fulltext index: {e}")

# Extract entities from text
class Entities(BaseModel):
    """Identifying information about entities."""

    names: List[str] = Field(
        ...,
        description="All the noun entities that "
        "appear in the text",
    )

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are extracting noun entities from the text. Don't include any explanation or text.",
        ),
        (
            "human",
            "Please extract all the noun entities into a list from the following "
            "input: {question}",
        ),
    ]
)

entity_chain = prompt | llm

def generate_full_text_query(input: str) -> str:
    """
    Generate a full-text search query for a given input string.

    This function constructs a query string suitable for a full-text search.
    It processes the input string by splitting it into words and appending a
    similarity threshold (~2 changed characters) to each word, then combines
    them using the AND operator. Useful for mapping entities from user questions
    to database values, and allows for some misspelings.
    """
    # print(input)
    full_text_query = ""
    words = [el for el in remove_lucene_chars(input).split() if el]
    if len(words) > 1:
        for word in words[:-1]:
            full_text_query += f" {word}~2 AND"
        full_text_query += f" {words[-1]}~2"
    else:
        full_text_query = f"{words[0]}~2"
    return full_text_query.strip()

# Fulltext index query
def structured_retriever(question: str, source_filter: str = None) -> str:
    """
    Collects the neighborhood of entities mentioned
    in the question, optionally filtered by source
    """
    result = ""
    entities = entity_chain.invoke({"question": question})
    # Add missing quotes around list elements
    fixed_content = re.sub(r'\[([^\'"\]]+)\]', r"['\1']", entities.content)
    list_entities = ast.literal_eval(fixed_content)
    for entity in list_entities:
        if source_filter:
            # Add source filter via Document nodes
            response = graph.query(
                """CALL db.index.fulltext.queryNodes('entity', $query, {limit:5})
                YIELD node,score
                CALL {
                  WITH node
                  MATCH (doc:Document)-[:MENTIONS]->(node)-[r:!MENTIONS]->(neighbor)
                  WHERE doc.source = $source OR doc.source CONTAINS $source
                  RETURN node.id + ' - ' + type(r) + ' -> ' + neighbor.id AS output
                  UNION ALL
                  WITH node
                  MATCH (doc:Document)-[:MENTIONS]->(node)<-[r:!MENTIONS]-(neighbor)
                  WHERE doc.source = $source OR doc.source CONTAINS $source
                  RETURN neighbor.id + ' - ' + type(r) + ' -> ' +  node.id AS output
                  UNION ALL
                  WITH node
                  MATCH (node)-[:MENTIONS]->(neighbor)
                  RETURN 'Entity: ' + node.id + ' - MENTIONS -> ' + neighbor.id AS output
                }
                RETURN output LIMIT 40
                """,
                {"query": generate_full_text_query(entity), "source": source_filter},
            )
        else:
            # Original query without source filter
            response = graph.query(
                """CALL db.index.fulltext.queryNodes('entity', $query, {limit:2})
                YIELD node,score
                CALL {
                  WITH node
                  MATCH (node)-[r:!MENTIONS]->(neighbor)
                  RETURN node.id + ' - ' + type(r) + ' -> ' + neighbor.id AS output
                  UNION ALL
                  WITH node
                  MATCH (node)<-[r:!MENTIONS]-(neighbor)
                  RETURN neighbor.id + ' - ' + type(r) + ' -> ' +  node.id AS output
                }
                RETURN output LIMIT 50
                """,
                {"query": generate_full_text_query(entity)},
            )
        result += "\n".join([el['output'] for el in response])
    return result

def retriever(question: str, source_filter: str = None):
    # print(f"Search query: {question}")
    structured_data = structured_retriever(question, source_filter)
    if vector_index is not None:
        if source_filter:
            # Filter by source if provided
            results = vector_index.similarity_search_with_score(question)
            unstructured_data = [
                doc.page_content for doc, score in results 
                if doc.metadata.get('source', '').lower() == source_filter.lower()
            ]
            # If no matching results, fall back to all results
            if not unstructured_data and results:
                unstructured_data = [el.page_content for el, _ in results[:3]]
        else:
            unstructured_data = [el.page_content for el in vector_index.similarity_search(question, k=5)]
    else:
        unstructured_data = []
    final_data = f"""Structured data:
{structured_data}
Unstructured data:
{"#Document ". join(unstructured_data)}
    """
    return final_data

# Condense a chat history and follow-up question into a standalone question
_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question,
in English.
Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""  # noqa: E501
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

def _format_chat_history(chat_history: List[Tuple[str, str]]) -> List:
    buffer = []
    for human, ai in chat_history:
        buffer.append(HumanMessage(content=human))
        buffer.append(AIMessage(content=ai))
    return buffer

_search_query = RunnableBranch(
    # If input includes chat_history, we condense it with the follow-up question
    (
        RunnableLambda(lambda x: bool(x.get("chat_history"))).with_config(
            run_name="HasChatHistoryCheck"
        ),  # Condense follow-up question and chat into a standalone_question
        RunnablePassthrough.assign(
            chat_history=lambda x: _format_chat_history(x["chat_history"])
        )
        | CONDENSE_QUESTION_PROMPT
        | llm
        | StrOutputParser(),
    ),
    # Else, we have no chat history, so just pass through the question
    RunnableLambda(lambda x : x["question"]),
)

if config["Chat"]["chatbot"].lower() == "ollama":
    from langchain_community.chat_models import ChatOllama
    # Lower temperature for more technical, factual answers
    chatbot = ChatOllama(model=config["Chat"]["model"], options={"temperature":float(config.get("Chat", "temperature", fallback="0.2"))})
elif config["Chat"]["chatbot"].lower() == "openai":
    from langchain_openai import ChatOpenAI
    chatbot = ChatOpenAI(
        # Lower temperature for more technical, factual answers
        temperature=float(config.get("Chat", "temperature", fallback="0.2")),
        max_tokens=int(config.get("Chat", "max_tokens", fallback="4096")),  # Increased for longer technical explanations
        openai_api_key=config.get("OpenAI", "api_key", fallback=""),
        openai_api_base=config.get("OpenAI", "api_base", fallback=None),
        stop=config.get("Chat", "stop", fallback=None),
    )
else:
    chatbot = llm

# Define prompts for different query types
qa_template = """You are answering questions about a research paper based on the following context extracted from the paper.

Context from the paper:
{context}

Question: {question}

Instructions:
- Provide a detailed, technical answer based on the context above
- ALWAYS include specific model names (e.g., ResNet, BERT, GPT-4, etc.) if mentioned
- Explain WHY each model/technique/method is used - don't just list names
- Include specific details about architectures, components, and how they work
- If discussing results, cite specific metrics, numbers, or performance measures
- Mention datasets used and their purposes
- Include training details, hyperparameters, or implementation specifics if available
- If the answer is not in the provided context, say "This information is not available in the provided context"
- Write in a professional, academic tone with specific technical details
- Do not make generic statements - always cite what's in the paper
- Structure your answer with specific sections when discussing methodology

Format your answer as follows:
1. If discussing models: List each model name and explain its purpose/role
2. If discussing methods: Explain the approach step-by-step with technical details
3. If discussing results: Cite specific numbers and comparisons

Answer:"""

hypothesis_template = """You are a research analyst generating novel hypotheses based on a research paper and related work.

Paper Context:
{context}

User Query: {question}

Your task is to generate research hypotheses that:
1. Build upon the findings, methods, or claims in the paper
2. Suggest testable research questions or experiments
3. Propose extensions or alternative approaches
4. Identify gaps or limitations that could be addressed

For each hypothesis, provide:
- **Hypothesis Statement**: A clear, testable prediction or research question
- **Rationale**: Why this hypothesis is plausible based on the paper's findings
- **Connection**: How it relates to or extends the paper's work
- **Testability**: How you could test this hypothesis (methodology, metrics, data requirements)

Generate 3-5 hypotheses in a structured format like this:

## Hypothesis 1: [Title]
**Statement**: [Your hypothesis]
**Rationale**: [Why this is plausible based on the paper]
**Connection to paper**: [How it extends/builds on the paper's work]
**Testability**: [How to test this]

Generate hypotheses now:"""

# Default to QA template
template = qa_template
prompt = ChatPromptTemplate.from_template(template)

# Chain that accepts source_filter as part of the input
def create_context(input_data):
    """Extract context using retriever with source_filter"""
    question = input_data.get("question", "")
    source_filter = input_data.get("source_filter")
    
    # Get context from Document nodes if source_filter is provided
    if source_filter:
        # Use vector similarity search if available, otherwise get related documents
        if vector_index:
            # Get similar documents to the question (increased k for more context)
            similar_docs = vector_index.similarity_search_with_score(
                question,
                k=12  # Increased from 8 for more comprehensive context
            )
            context = f"Relevant sections from {source_filter}:\n\n"
            for i, (doc, score) in enumerate(similar_docs, 1):
                # Filter by source and only include relevant ones
                if doc.metadata.get('source', '').lower() in source_filter.lower():
                    context += f"--- Section {i} (relevance: {score:.2f}) ---\n"
                    context += f"{doc.page_content}\n\n"
        else:
            # Fallback: Get all documents and include them
            doc_results = graph.query("""
            MATCH (d:Document)
            WHERE (d.source = $source OR d.source CONTAINS $source) AND d.text IS NOT NULL
            RETURN d.text as text
            LIMIT 20
            """, {"source": source_filter})
            
            context = f"Content from {source_filter}:\n\n"
            for doc in doc_results:
                # Include full text, not truncated
                context += f"{doc['text']}\n\n---\n\n"
        
        # Also add entity information for better context, especially models, methods, and techniques
        entity_results = graph.query("""
        MATCH (d:Document)-[:MENTIONS]->(e:__Entity__)
        WHERE (d.source = $source OR d.source CONTAINS $source)
        WITH e, count(*) as freq
        ORDER BY freq DESC
        RETURN e.id as entity, freq
        LIMIT 20
        """, {"source": source_filter})
        
        if entity_results:
            context += "\n\nKey technical entities and concepts mentioned in this paper:\n"
            for e in entity_results:
                context += f"- {e['entity']} (mentioned {e['freq']} times)\n"
        
        # Also try to get sections that specifically mention methodology or experiments
        methodology_docs = graph.query("""
        MATCH (d:Document)
        WHERE (d.source = $source OR d.source CONTAINS $source) 
          AND d.text IS NOT NULL
          AND (
            toLower(d.text) CONTAINS 'method' OR 
            toLower(d.text) CONTAINS 'model' OR 
            toLower(d.text) CONTAINS 'architecture' OR 
            toLower(d.text) CONTAINS 'approach' OR
            toLower(d.text) CONTAINS 'experiment'
          )
        RETURN d.text as text
        LIMIT 5
        """, {"source": source_filter})
        
        if methodology_docs:
            context += "\n\n--- Methodology Sections ---\n"
            for doc in methodology_docs:
                context += f"{doc['text'][:1000]}\n\n"
    else:
        # Fallback to original retriever
        context = retriever(question, source_filter)
    
    return {
        "context": context,
        "question": question
    }

def create_chain(query_type="qa"):
    """Create a chain with the specified prompt template"""
    global prompt, template
    
    if query_type == "hypothesis":
        template = hypothesis_template
    else:
        template = qa_template
    
    prompt = ChatPromptTemplate.from_template(template)
    
    return (
        RunnableLambda(create_context)
        | prompt
        | chatbot
        | StrOutputParser()
    )

# Default chain for QA
chain = create_chain("qa")

if __name__ == "__main__":
    while True:
        human_input = input("\n>>>")
        if human_input == "exit":
            break
        for chunk in chain.stream({"question": human_input}):
            print(chunk, end="", flush=True)