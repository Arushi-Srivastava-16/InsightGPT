"""
InsightGPT - Single Page Version
All features on one scrollable page
"""
import streamlit as st
import os
import tempfile
from pathlib import Path
from datetime import datetime
import sys
import pandas as pd
import networkx as nx
from pyvis.network import Network
import plotly.express as px
import plotly.graph_objects as go

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.summarizer import ResearchSummarizer
from langchain_neo4j import Neo4jGraph
from src.core.citation_validator import CitationValidator
from src.utils.config_loader import load_config as load_env_config

# Page config
st.set_page_config(
    page_title="InsightGPT - Research Copilot",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .section-header {
        font-size: 2rem;
        color: #667eea;
        border-bottom: 3px solid #667eea;
        padding-bottom: 10px;
        margin-top: 40px;
        margin-bottom: 20px;
    }
    .stButton>button {
        border-radius: 8px;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Initialize components
@st.cache_resource
def load_config():
    return load_env_config(use_env=True)

@st.cache_resource
def initialize_components():
    try:
        config = load_config()
        os.environ["NEO4J_URI"] = config["Neo4j"]["uri"]
        os.environ["NEO4J_USERNAME"] = config["Neo4j"]["username"]
        os.environ["NEO4J_PASSWORD"] = config["Neo4j"]["password"]
        
        summarizer = ResearchSummarizer()
        citation_validator = CitationValidator()
        
        # Test Neo4j connection explicitly
        try:
            graph = Neo4jGraph()
            # Test with a simple query
            graph.query("RETURN 1 as test")
            print("SUCCESS: Neo4j connection successful")
        except Exception as neo4j_error:
            print(f"WARNING: Neo4j connection issue: {neo4j_error}")
            graph = None
        
        return summarizer, citation_validator, graph, config
    except Exception as e:
        print(f"ERROR: Initialization error: {e}")
        raise e

# Session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'processed_papers' not in st.session_state:
    st.session_state.processed_papers = []
if 'current_source' not in st.session_state:
    st.session_state.current_source = None

# Initialize
summarizer = citation_validator = graph = config = None

try:
    summarizer, citation_validator, graph, config = initialize_components()
    # If we get here, initialization was successful
    if graph is None:
        st.warning("WARNING: Neo4j connection failed, but app will work in limited mode")
    else:
        st.success("SUCCESS: All components initialized successfully!")
except Exception as e:
    # More graceful error handling
    error_msg = str(e)
    st.error(f"ERROR initializing: {error_msg}")
    
    # Try to at least get config
    try:
        config = load_config()
    except:
        config = None
    
    # Set components to None for graceful degradation
    summarizer = None
    citation_validator = None
    graph = None

# Sidebar - Quick Navigation
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/artificial-intelligence.png", width=80)
    st.markdown("# 🔬 InsightGPT")
    st.markdown("*AI-Powered Research Copilot*")
    st.markdown("---")
    
    st.markdown("### 📍 Quick Navigation")
    st.markdown("""
    - [🏠 Welcome](#welcome)
    - [📄 Upload PDF](#upload-pdf)
    - [💬 Chat & Query](#chat-query)
    - [📚 Summarize](#summarize)
    - [🧠 Hypotheses](#hypotheses)
    - [📊 Visualizations](#visualizations)
    - [📖 Citations](#citations)
    """)
    
    st.markdown("---")
    st.markdown("### 📊 Stats")
    if graph:
        try:
            doc_count = graph.query("MATCH (d:Document) RETURN count(d) as count")
            entity_count = graph.query("MATCH (e:__Entity__) RETURN count(e) as count")
            st.metric("Documents", doc_count[0]['count'] if doc_count else 0)
            st.metric("Entities", entity_count[0]['count'] if entity_count else 0)
        except Exception as e:
            st.warning("WARNING: Neo4j connection issue - stats unavailable")
    st.metric("Papers Processed", len(st.session_state.processed_papers))

# =======================
# MAIN CONTENT - ALL SECTIONS
# =======================

# Welcome Section
st.markdown('<p class="main-header">🔬 InsightGPT</p>', unsafe_allow_html=True)
st.markdown("### Your AI-Powered Research Copilot")
st.markdown("All features on one page - scroll down to explore!")

col1, col2, col3 = st.columns(3)
with col1:
    st.info("📄 **PDF Processing**\nExtract entities & build graphs")
with col2:
    st.info("💬 **Intelligent Q&A**\nHybrid graph + vector search")
with col3:
    st.info("🧠 **Research Insights**\nSummaries & hypotheses")

st.markdown("---")

# Section 1: Upload PDF
st.markdown('<div id="upload-pdf"></div>', unsafe_allow_html=True)
st.markdown('<p class="section-header">📄 Upload & Process PDF</p>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("Choose a PDF file", type=['pdf'])

if uploaded_file:
    st.success(f"✅ File: {uploaded_file.name} ({uploaded_file.size / 1024:.2f} KB)")
    
    extract_images = st.checkbox("Extract images", value=False)
    
    if st.button("🚀 Process PDF", key="process_btn"):
        # Create a progress bar
        status_text = st.empty()
        progress_bar = st.progress(0)
        
        with st.spinner("Processing... This may take a few minutes"):
            try:
                from pdf2graph import process_document
                
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_path = tmp_file.name
                
                progress_bar = st.progress(0)
                progress_bar.progress(25)
                
                metadata = {
                    "source": uploaded_file.name,
                    "upload_date": datetime.now().isoformat()
                }
                
                progress_bar.progress(50)
                
                # Delete old pickle file if it exists to force re-processing
                if os.path.exists("output.pkl"):
                    os.remove("output.pkl")
                
                # Process the document
                # Use larger chunks to reduce total number of documents
                # This will make processing much faster (fewer LLM calls)
                process_document(
                    tmp_path,
                    metadata,
                    images=extract_images,
                    max_char=int(config["PDF"]["max_char"]) if config else 3000,  # Increased from 1000 to 3000
                    new_after_n_chars=int(config["PDF"]["new_after_n_chars"]) if config else 2400,  # Increased from 800 to 2400
                    combine=int(config["PDF"]["combine_text_under_n_chars"]) if config else 200
                )
                
                progress_bar.progress(100)
                status_text.text("✅ Processing complete!")
                os.unlink(tmp_path)
                
                st.success(f"🎉 Successfully processed: {uploaded_file.name}")
                st.session_state.processed_papers.append(uploaded_file.name)
                st.session_state.current_source = uploaded_file.name
                # Clear chat history when processing a new paper
                st.session_state.chat_history = []
                st.balloons()
                st.rerun()
                
            except Exception as e:
                error_msg = str(e)
                if "Neo4j" in error_msg:
                    st.error(f"ERROR: Neo4j database issue - {error_msg}")
                    st.info("TIP: Make sure Neo4j Desktop is running and the database is accessible")
                else:
                    st.error(f"ERROR: {error_msg}")

st.markdown("---")

# Section 2: Chat & Query
st.markdown('<div id="chat-query"></div>', unsafe_allow_html=True)
st.markdown('<p class="section-header">💬 Chat & Query</p>', unsafe_allow_html=True)

# Paper selector
if st.session_state.processed_papers:
    st.session_state.current_source = st.selectbox(
        "📄 Select Paper to Discuss",
        options=st.session_state.processed_papers,
        index=st.session_state.processed_papers.index(st.session_state.current_source) if st.session_state.current_source in st.session_state.processed_papers else 0,
        key="paper_selector"
    )
    
    # Show current paper
    st.info(f"📄 Currently discussing: **{st.session_state.current_source}**")
    
    # Add a button to clear chat history when switching papers
    if st.button("🔄 Clear Chat History", key="clear_chat_btn"):
        st.session_state.chat_history = []
        st.rerun()

# Display chat history
for msg in st.session_state.chat_history[-10:]:  # Show last 10 messages
    if msg['role'] == 'user':
        st.markdown(f"**🧑 You:** {msg['content']}")
    else:
        st.markdown(f"**🤖 InsightGPT:** {msg['content']}")

# Query input
query = st.text_input("Ask a question", key="query_input", placeholder="What is the main contribution of the paper?")

col1, col2, col3 = st.columns([1, 1, 4])
with col1:
    if st.button("🚀 Ask", key="ask_btn"):
        if query:
            st.session_state.chat_history.append({'role': 'user', 'content': query})
            
            try:
                from graphQA import chain as qa_chain
                # Pass the current source to filter results
                response = ""
                input_data = {
                    "question": query,
                    "source_filter": st.session_state.current_source
                }
                for chunk in qa_chain.stream(input_data):
                    response += chunk
                
                st.session_state.chat_history.append({'role': 'assistant', 'content': response})
                st.rerun()
            except Exception as e:
                st.error(f"ERROR: {str(e)}")

with col2:
    if st.button("💡 Generate Hypotheses", key="hypothesis_btn"):
        if query:
            st.session_state.chat_history.append({'role': 'user', 'content': f"Generate hypotheses: {query}"})
            
            try:
                from graphQA import create_chain
                # Create a hypothesis generation chain
                hyp_chain = create_chain("hypothesis")
                
                response = ""
                input_data = {
                    "question": query,
                    "source_filter": st.session_state.current_source
                }
                for chunk in hyp_chain.stream(input_data):
                    response += chunk
                
                st.session_state.chat_history.append({'role': 'assistant', 'content': response})
                st.rerun()
            except Exception as e:
                st.error(f"ERROR: {str(e)}")

with col3:
    if st.button("🗑️ Clear History", key="clear_btn"):
        st.session_state.chat_history = []
        st.rerun()

st.markdown("---")

# Section 3: Summarize
st.markdown('<div id="summarize"></div>', unsafe_allow_html=True)
st.markdown('<p class="section-header">📚 Document Summarization</p>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    summary_input = st.text_input("Enter paper title (optional)", key="summary_input", placeholder="Leave empty to summarize all")
with col2:
    summarize_all = st.button("📝 Summarize All", key="summarize_all_btn", type="primary")

if st.button("📝 Generate Summary", key="summary_btn") or summarize_all:
    if summarizer:
        with st.spinner("Generating summary..."):
            try:
                if summarize_all or not summary_input:
                    # Summarize all documents
                    summary = summarizer.summarize_from_graph(paper_title=None)
                else:
                    # Summarize specific paper
                    summary = summarizer.summarize_from_graph(paper_title=summary_input)
                
                st.markdown("### 📋 Summary")
                st.info(summary)
                
                filename = "summary_all.txt" if summarize_all or not summary_input else f"summary_{summary_input.replace(' ', '_')}.txt"
                st.download_button(
                    "📥 Download",
                    summary,
                    file_name=filename
                )
            except Exception as e:
                st.error(f"ERROR: {str(e)}")
    else:
        st.warning("Components not loaded. Please restart the app.")

st.markdown("---")

# Section 4: Generate Hypotheses
st.markdown('<div id="hypotheses"></div>', unsafe_allow_html=True)
st.markdown('<p class="section-header">🧠 Hypothesis Generation</p>', unsafe_allow_html=True)

context = st.text_area("Enter research context", height=150, key="hyp_context")
research_q = st.text_input("Research question (optional)", key="research_q")

if st.button("🚀 Generate Hypotheses", key="hyp_btn"):
    if context and summarizer:
        with st.spinner("Generating..."):
            try:
                hypotheses = summarizer.generate_hypothesis(context, research_q if research_q else None)
                st.markdown("### 🎯 Generated Hypotheses")
                st.success(hypotheses)
                
                st.download_button(
                    "📥 Download",
                    hypotheses,
                    file_name="hypotheses.txt"
                )
            except Exception as e:
                st.error(f"ERROR: {str(e)}")
    else:
        st.warning("Please provide context")

st.markdown("---")

# Section 5: Citations
st.markdown('<div id="citations"></div>', unsafe_allow_html=True)
st.markdown('<p class="section-header">📖 Citation Management</p>', unsafe_allow_html=True)

citation_text = st.text_area("Paste text with citations", height=150, key="cite_text")
citation_style = st.selectbox("Citation Style", ["apa", "mla", "chicago", "harvard"], key="cite_style")

col_validate1, col_validate2 = st.columns([1, 1])

with col_validate1:
    if st.button("✅ Validate Citations", key="cite_btn"):
        if citation_text and citation_validator:
            with st.spinner("Validating..."):
                try:
                    report = citation_validator.batch_validate_citations(citation_text, citation_style)
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total", report['total_citations'])
                    with col2:
                        st.metric("Valid", report['valid_count'])
                    with col3:
                        st.metric("Accuracy", f"{report['accuracy']:.1f}%")
                    
                    if report['issues']:
                        st.warning("⚠️ Issues Found:")
                        for issue in report['issues']:
                            st.markdown(f"- {issue}")
                    else:
                        st.success("✅ All citations properly formatted!")
                    
                    # Show detailed validations
                    if report['validations']:
                        st.markdown("### 📋 Detailed Results")
                        for i, val in enumerate(report['validations'], 1):
                            status = "✅" if val['format_valid'] else "❌"
                            st.markdown(f"{status} **Citation {i}**: `{val['citation']}`")
                            if not val['format_valid']:
                                st.markdown(f"   - Issue: {val.get('format_issues', 'Unknown')}")
                                if val.get('suggestion'):
                                    st.markdown(f"   - Suggestion: {val['suggestion']}")
                            if val.get('exists_in_graph'):
                                st.markdown(f"   - ✅ Found in knowledge graph")
                            else:
                                st.markdown(f"   - ⚠️ Not found in knowledge graph")
                                
                except Exception as e:
                    st.error(f"ERROR: {str(e)}")

with col_validate2:
    if st.button("🔍 Check in Knowledge Graph", key="check_cite_btn"):
        if citation_text and citation_validator:
            with st.spinner("Checking knowledge graph..."):
                try:
                    citations = citation_validator.extract_citations_from_text(citation_text)
                    
                    if citations:
                        st.markdown(f"### Found {len(citations)} citation(s)")
                        
                        for i, citation in enumerate(citations, 1):
                            with st.expander(f"Citation {i}: `{citation['raw']}`"):
                                exists_check = citation_validator.check_citation_exists(citation['raw'])
                                
                                if exists_check['exists']:
                                    st.success(f"✅ Found {exists_check['count']} match(es) in graph")
                                    for match in exists_check.get('matches', []):
                                        st.markdown(f"- **{match.get('citation', 'Unknown')}**")
                                        if match.get('source'):
                                            st.markdown(f"  Source: {match['source']}")
                                else:
                                    st.warning("⚠️ Citation not found in knowledge graph")
                                    st.info("💡 Upload the cited paper to add it to the knowledge graph")
                    else:
                        st.info("No citations found in the text")
                        
                except Exception as e:
                    st.error(f"ERROR: {str(e)}")

# Section 6: Visualizations
st.markdown("---")
st.markdown('<div id="visualizations"></div>', unsafe_allow_html=True)
st.markdown('<p class="section-header">📊 Knowledge Graph Visualizations</p>', unsafe_allow_html=True)

if graph:
    try:
        # Query statistics for visualizations
        node_counts = graph.query("""
        MATCH (n:Document)
        RETURN count(n) as count, 'Document' as label
        UNION ALL
        MATCH (n:__Entity__)
        RETURN count(n) as count, '__Entity__' as label
        """)
        
        if node_counts:
            st.markdown("### 📈 Graph Statistics")
            
            # Create DataFrame
            df_nodes = pd.DataFrame(node_counts)
            
            # Plot bar chart
            fig = px.bar(
                df_nodes, 
                x='label', 
                y='count',
                title='Node Counts by Type',
                labels={'label': 'Node Type', 'count': 'Count'},
                color='count',
                color_continuous_scale='viridis'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Query for relationship statistics
        rel_counts = graph.query("""
        MATCH ()-[r]->()
        RETURN type(r) as relationshipType, count(*) as count
        ORDER BY count DESC
        LIMIT 10
        """)
        
        if rel_counts:
            df_rels = pd.DataFrame(rel_counts)
            
            fig = px.pie(
                df_rels, 
                values='count', 
                names='relationshipType',
                title='Relationship Distribution',
                color_discrete_sequence=px.colors.sequential.Viridis
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        # Network graph visualization
        st.markdown("### 🕸️ Knowledge Graph Network")
        
        col_viz1, col_viz2 = st.columns(2)
        
        with col_viz1:
            depth = st.slider("Graph Depth", 1, 3, 1, key="viz_depth")
        
        with col_viz2:
            max_nodes = st.number_input("Max Nodes", 10, 100, 30, key="max_nodes")
        
        if st.button("🔄 Generate Graph", key="gen_graph"):
            try:
                with st.spinner("Querying knowledge graph..."):
                    # Query graph relationships - use only properties that exist
                    query = f"""
                    MATCH (start)-[r]-(end)
                    WHERE (start:__Entity__ OR start:Document) AND (end:__Entity__ OR end:Document)
                    WITH start, end, type(r) as rel_type
                    ORDER BY coalesce(start.id, '') DESC
                    RETURN DISTINCT
                       coalesce(start.id, left(toString(id(start)), 20)) as start_name,
                       coalesce(end.id, left(toString(id(end)), 20)) as end_name,
                       labels(start)[0] as start_label,
                       labels(end)[0] as end_label,
                       coalesce(start.text, '') as start_text,
                       coalesce(end.text, '') as end_text,
                       coalesce(start.source, '') as start_source,
                       coalesce(end.source, '') as end_source,
                       rel_type as rel_type,
                       1 as weight
                    LIMIT {max_nodes}
                    """
                    
                    result = graph.query(query)
                    
                    if result:
                        # Create networkx graph
                        G = nx.Graph()
                        
                        for row in result:
                            if row['start_name'] and row['end_name']:
                                G.add_edge(
                                    row['start_name'], 
                                    row['end_name'],
                                    weight=row.get('weight', 1),
                                    start_label=row['start_label'],
                                    end_label=row['end_label']
                                )
                        
                        if G.number_of_nodes() > 0:
                            # Create Pyvis network
                            net = Network(
                                height='600px',
                                width='100%',
                                bgcolor='#0e1117',
                                font_color='white'
                            )
                            
                            # Add nodes with colors by label
                            color_map = {
                                'Document': '#ff6b6b',
                                '__Entity__': '#4ecdc4',
                                'Paper': '#95e1d3',
                            }
                            
                            # Create a mapping of node names to their metadata
                            node_metadata = {}
                            for row in result:
                                start_node = row['start_name']
                                end_node = row['end_name']
                                
                                if start_node and start_node not in node_metadata:
                                    text = row.get('start_text', '')
                                    source = row.get('start_source', '')
                                    
                                    # Prioritize: text content > source > id
                                    if text and len(text) > 3:
                                        display_name = text[:50] if len(text) > 50 else text
                                    elif source:
                                        display_name = source[:30]
                                    else:
                                        display_name = start_node if len(str(start_node)) < 30 else str(start_node)[:30]
                                    
                                    node_metadata[start_node] = {
                                        'label': row['start_label'],
                                        'display_name': display_name,
                                        'text': text[:100] if text else ''
                                    }
                                
                                if end_node and end_node not in node_metadata:
                                    text = row.get('end_text', '')
                                    source = row.get('end_source', '')
                                    
                                    if text and len(text) > 3:
                                        display_name = text[:50] if len(text) > 50 else text
                                    elif source:
                                        display_name = source[:30]
                                    else:
                                        display_name = end_node if len(str(end_node)) < 30 else str(end_node)[:30]
                                    
                                    node_metadata[end_node] = {
                                        'label': row['end_label'],
                                        'display_name': display_name,
                                        'text': text[:100] if text else ''
                                    }
                            
                            for node in G.nodes():
                                metadata = node_metadata.get(node, {})
                                node_label = metadata.get('label', 'Entity')
                                display_name = metadata.get('display_name', str(node)[:30])
                                text_preview = metadata.get('text', str(node))
                                
                                # Create tooltip with full information
                                tooltip = f"{node_label}\nID: {str(node)}\n\n{text_preview[:200]}"
                                
                                color = color_map.get(node_label, '#667eea')
                                net.add_node(node, label=display_name, color=color, title=tooltip)
                            
                            # Add edges
                            for edge in G.edges(data=True):
                                net.add_edge(edge[0], edge[1], weight=edge[2].get('weight', 1))
                            
                            # Configure physics
                            net.set_options("""
                            {
                              "physics": {
                                "enabled": true,
                                "stabilization": {"iterations": 100},
                                "forceAtlas2Based": {
                                  "gravitationalConstant": -50,
                                  "centralGravity": 0.02,
                                  "springLength": 100,
                                  "springConstant": 0.08
                                }
                              }
                            }
                            """)
                            
                            # Save and show graph using a better approach
                            import uuid
                            temp_dir = tempfile.gettempdir()
                            html_filename = os.path.join(temp_dir, f"graph_{uuid.uuid4().hex}.html")
                            net.save_graph(html_filename)
                            
                            # Read the HTML content
                            with open(html_filename, 'r', encoding='utf-8') as f:
                                html_content = f.read()
                            
                            # Display the graph
                            st.components.v1.html(html_content, height=600, scrolling=True)
                            
                            # Clean up - use try/except to avoid file lock issues
                            try:
                                os.remove(html_filename)
                            except:
                                pass  # Ignore deletion errors
                        else:
                            st.info("No graph data found. Upload some papers first!")
                    else:
                        st.warning("No relationships found in the knowledge graph.")
                        
            except Exception as e:
                st.error(f"ERROR generating graph: {str(e)}")
        
        # Entity co-occurrence heatmap
        st.markdown("### 🔗 Entity Co-occurrence")
        
        if st.button("📊 Show Co-occurrence", key="cooccur"):
            try:
                with st.spinner("Analyzing entity co-occurrences..."):
                    cooccur_query = """
                    MATCH (d:Document)-[:MENTIONS]->(e1:__Entity__)<-[:MENTIONS]-(d)-[:MENTIONS]->(e2:__Entity__)
                    WHERE id(e1) < id(e2)
                    RETURN e1.id as entity1, e2.id as entity2, count(*) as cooccurrence
                    ORDER BY cooccurrence DESC
                    LIMIT 100
                    """
                    
                    cooccur_data = graph.query(cooccur_query)
                    
                    if cooccur_data:
                        df_cooccur = pd.DataFrame(cooccur_data)
                        
                        # Create pivot table
                        pivot = df_cooccur.pivot_table(
                            index='entity1', 
                            columns='entity2', 
                            values='cooccurrence',
                            fill_value=0
                        )
                        
                        # Plot heatmap
                        fig = px.imshow(
                            pivot.values,
                            labels=dict(x="Entity 2", y="Entity 1", color="Co-occurrence"),
                            x=pivot.columns,
                            y=pivot.index,
                            color_continuous_scale='viridis',
                            aspect='auto'
                        )
                        fig.update_layout(height=600)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No co-occurrence data found.")
                        
            except Exception as e:
                st.error(f"ERROR: {str(e)}")
                
    except Exception as e:
        st.warning(f"WARNING: Could not load visualizations: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem;'>
    <p>🔬 InsightGPT - AI-Powered Research Copilot</p>
    <p>Built with Streamlit • LangChain • Neo4j • OpenAI/Ollama</p>
</div>
""", unsafe_allow_html=True)

