"""
GPT Citation Validator for InsightGPT
Validates citation accuracy and formatting
"""
import os
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.utils.config_loader import load_config as load_env_config, get_llm_config
from langchain_neo4j import Neo4jGraph

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
            config = get_config()
            # Set environment variables for Neo4j
            os.environ["NEO4J_URI"] = config["Neo4j"]["uri"]
            os.environ["NEO4J_USERNAME"] = config["Neo4j"]["username"]
            os.environ["NEO4J_PASSWORD"] = config["Neo4j"]["password"]
            _graph = Neo4jGraph()
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
                "max_tokens": int(llm_cfg.get("max_tokens", 2048)),
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
            raise ValueError("Invalid LLM model configuration: provider should be 'openai' or 'ollama'")
    return _llm


class CitationValidator:
    """
    Validates citations for accuracy, formatting, and context appropriateness
    """
    
    def __init__(self, llm=None):
        self.llm = llm or get_llm()
        self.graph = get_graph()
        
        # Citation format patterns
        self.citation_patterns = {
            'apa': r'\(([A-Za-z\s,&]+),\s*(\d{4})\)',
            'mla': r'([A-Za-z\s]+),\s*([A-Za-z\s]+)\.',
            'chicago': r'([A-Za-z\s]+),\s*"([^"]+),"',
            'numeric': r'\[(\d+)\]',
            'harvard': r'\(([A-Za-z\s,&]+)\s+(\d{4})\)'
        }
    
    def extract_citations_from_text(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract all citations from text using pattern matching
        
        Args:
            text: Text containing citations
            
        Returns:
            List of extracted citations with metadata
        """
        citations = []
        
        for style, pattern in self.citation_patterns.items():
            matches = re.finditer(pattern, text)
            for match in matches:
                citation = {
                    'raw': match.group(0),
                    'style': style,
                    'position': match.span(),
                    'groups': match.groups()
                }
                
                # Extract author and year for APA/Harvard
                if style in ['apa', 'harvard'] and len(match.groups()) >= 2:
                    citation['author'] = match.groups()[0].strip()
                    citation['year'] = match.groups()[1].strip()
                
                citations.append(citation)
        
        return citations
    
    def validate_citation_format(self, citation: str, expected_style: str = 'apa') -> Dict[str, Any]:
        """
        Validate if citation follows proper format
        
        Args:
            citation: Citation string to validate
            expected_style: Expected citation style (apa, mla, chicago, etc.)
            
        Returns:
            Validation result with issues
        """
        pattern = self.citation_patterns.get(expected_style)
        
        if not pattern:
            return {
                'valid': False,
                'error': f'Unknown citation style: {expected_style}'
            }
        
        match = re.search(pattern, citation)
        
        if match:
            return {
                'valid': True,
                'style': expected_style,
                'components': match.groups()
            }
        else:
            return {
                'valid': False,
                'style': expected_style,
                'error': f'Citation does not match {expected_style} format',
                'suggestion': self.suggest_format(citation, expected_style)
            }
    
    def suggest_format(self, citation: str, style: str) -> str:
        """
        Suggest proper formatting for a citation
        
        Args:
            citation: Malformed citation
            style: Target citation style
            
        Returns:
            Suggested formatted citation
        """
        # Use LLM to suggest proper format
        prompt = PromptTemplate(
            template="""You are a citation formatting expert.

Given this citation: {citation}

Format it properly in {style} style.
Provide ONLY the formatted citation, no explanation.

Formatted citation:""",
            input_variables=["citation", "style"]
        )
        
        chain = prompt | self.llm | StrOutputParser()
        suggestion = chain.invoke({"citation": citation, "style": style})
        
        return suggestion.strip()
    
    def validate_citation_context(self, citation: str, context: str, paper_content: str = None) -> Dict[str, Any]:
        """
        Validate if citation is used appropriately in context
        
        Args:
            citation: The citation being checked
            context: Surrounding text where citation appears
            paper_content: Full content of cited paper (if available)
            
        Returns:
            Validation result
        """
        validation_prompt = PromptTemplate(
            template="""You are validating citation usage in academic writing.

Citation: {citation}
Context where it's used: {context}
{paper_info}

Analyze:
1. Is the citation relevant to the context?
2. Is it being used appropriately (support, comparison, contrast)?
3. Are there any red flags (misrepresentation, out of context)?

Provide assessment:
- Valid: Yes/No
- Confidence: 0-100%
- Reasoning: Brief explanation
- Issues: Any problems found

Assessment:""",
            input_variables=["citation", "context", "paper_info"]
        )
        
        paper_info = f"Paper content: {paper_content[:500]}..." if paper_content else "Paper content not available."
        
        chain = validation_prompt | self.llm | StrOutputParser()
        assessment = chain.invoke({
            "citation": citation,
            "context": context,
            "paper_info": paper_info
        })
        
        # Parse assessment
        is_valid = 'valid: yes' in assessment.lower()
        
        return {
            'citation': citation,
            'context_valid': is_valid,
            'assessment': assessment,
            'timestamp': str(Path(__file__).parent)
        }
    
    def check_citation_exists(self, citation: str) -> Dict[str, Any]:
        """
        Check if citation exists in knowledge graph
        
        Args:
            citation: Citation to check
            
        Returns:
            Information about citation in graph
        """
        # Extract potential author and year
        authors = []
        year = None
        
        # Try to extract from different formats
        for pattern in self.citation_patterns.values():
            match = re.search(pattern, citation)
            if match:
                groups = match.groups()
                if len(groups) >= 1:
                    authors.append(groups[0])
                if len(groups) >= 2 and groups[1].isdigit():
                    year = groups[1]
                break
        
        # Query Neo4j for matching citations or papers
        if authors:
            query = """
            MATCH (c:Citation)
            WHERE c.text CONTAINS $author
            RETURN c.text as citation, c.source as source
            LIMIT 5
            """
            results = graph.query(query, {"author": authors[0]})
            
            if results:
                return {
                    'exists': True,
                    'matches': results,
                    'count': len(results)
                }
        
        # Also check in Papers
        if authors:
            query = """
            MATCH (p:Paper)
            WHERE p.title CONTAINS $author OR p.source CONTAINS $author
            RETURN p.title as title, p.source as source
            LIMIT 5
            """
            results = graph.query(query, {"author": authors[0]})
            
            if results:
                return {
                    'exists': True,
                    'matches': results,
                    'count': len(results),
                    'type': 'paper'
                }
        
        return {
            'exists': False,
            'message': 'Citation not found in knowledge graph'
        }
    
    def validate_bibliography_entry(self, entry: str, style: str = 'apa') -> Dict[str, Any]:
        """
        Validate a bibliography/reference entry
        
        Args:
            entry: Bibliography entry to validate
            style: Citation style to validate against
            
        Returns:
            Validation result with corrections
        """
        validation_prompt = PromptTemplate(
            template="""You are a bibliography validation expert.

Validate this {style} bibliography entry:
{entry}

Check for:
1. Proper formatting (author, year, title, journal/venue, etc.)
2. Punctuation and capitalization
3. Completeness (all required fields present)

Provide:
- Valid: Yes/No
- Issues: List any problems
- Corrected version: If needed

Analysis:""",
            input_variables=["entry", "style"]
        )
        
        chain = validation_prompt | self.llm | StrOutputParser()
        analysis = chain.invoke({"entry": entry, "style": style})
        
        is_valid = 'valid: yes' in analysis.lower()
        
        return {
            'entry': entry,
            'style': style,
            'valid': is_valid,
            'analysis': analysis
        }
    
    def generate_citation_from_metadata(self, metadata: Dict[str, Any], style: str = 'apa') -> str:
        """
        Generate properly formatted citation from metadata
        
        Args:
            metadata: Paper metadata (title, authors, year, venue, etc.)
            style: Target citation style
            
        Returns:
            Formatted citation string
        """
        generation_prompt = PromptTemplate(
            template="""Generate a properly formatted {style} citation from this metadata:

Title: {title}
Authors: {authors}
Year: {year}
Venue: {venue}
DOI: {doi}
Pages: {pages}

Provide ONLY the formatted citation:""",
            input_variables=["style", "title", "authors", "year", "venue", "doi", "pages"]
        )
        
        chain = generation_prompt | self.llm | StrOutputParser()
        citation = chain.invoke({
            "style": style,
            "title": metadata.get('title', 'Unknown'),
            "authors": metadata.get('authors', 'Unknown'),
            "year": metadata.get('year', 'n.d.'),
            "venue": metadata.get('venue', ''),
            "doi": metadata.get('doi', ''),
            "pages": metadata.get('pages', '')
        })
        
        return citation.strip()
    
    def batch_validate_citations(self, text: str, style: str = 'apa') -> Dict[str, Any]:
        """
        Validate all citations in a text
        
        Args:
            text: Text containing multiple citations
            style: Expected citation style
            
        Returns:
            Comprehensive validation report
        """
        # Extract all citations
        citations = self.extract_citations_from_text(text)
        
        results = {
            'total_citations': len(citations),
            'style': style,
            'validations': [],
            'issues': [],
            'valid_count': 0
        }
        
        for citation in citations:
            # Validate format
            format_check = self.validate_citation_format(citation['raw'], style)
            
            # Check if exists in graph
            exists_check = self.check_citation_exists(citation['raw'])
            
            validation = {
                'citation': citation['raw'],
                'format_valid': format_check['valid'],
                'exists_in_graph': exists_check['exists'],
                'style': citation['style']
            }
            
            if not format_check['valid']:
                validation['format_issues'] = format_check.get('error')
                validation['suggestion'] = format_check.get('suggestion')
                results['issues'].append(f"Format issue: {citation['raw']}")
            else:
                results['valid_count'] += 1
            
            results['validations'].append(validation)
        
        results['accuracy'] = (results['valid_count'] / results['total_citations'] * 100) if results['total_citations'] > 0 else 0
        
        return results


class AutoCiter:
    """
    Automatically generate and insert citations
    """
    
    def __init__(self, validator: CitationValidator = None):
        self.validator = validator or CitationValidator()
        self.llm = self.validator.llm
    
    def suggest_citations(self, text: str, paper_database: List[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Suggest where citations should be added in text
        
        Args:
            text: Text to analyze
            paper_database: Available papers to cite
            
        Returns:
            List of citation suggestions
        """
        suggestion_prompt = PromptTemplate(
            template="""Analyze this text and identify where citations are needed:

{text}

For each claim or statement that needs citation:
1. Identify the specific sentence or phrase
2. Explain why it needs citation
3. Suggest what type of paper to cite

Suggestions:""",
            input_variables=["text"]
        )
        
        chain = suggestion_prompt | self.llm | StrOutputParser()
        suggestions = chain.invoke({"text": text})
        
        return {
            'text': text,
            'suggestions': suggestions,
            'needs_citations': 'citation' in suggestions.lower() or 'cite' in suggestions.lower()
        }
    
    def auto_generate_intext_citation(self, claim: str, paper_metadata: Dict[str, Any], style: str = 'apa') -> str:
        """
        Generate in-text citation for a specific claim
        
        Args:
            claim: The claim being made
            paper_metadata: Metadata of paper to cite
            style: Citation style
            
        Returns:
            Formatted in-text citation
        """
        if style == 'apa' or style == 'harvard':
            authors = paper_metadata.get('authors', 'Unknown')
            year = paper_metadata.get('year', 'n.d.')
            
            # Simplify author list
            if ',' in authors:
                author_list = authors.split(',')
                if len(author_list) > 2:
                    first_author = author_list[0].split()[-1]  # Last name
                    return f"({first_author} et al., {year})"
                elif len(author_list) == 2:
                    return f"({author_list[0].split()[-1]} & {author_list[1].split()[-1]}, {year})"
                else:
                    return f"({author_list[0].split()[-1]}, {year})"
            else:
                return f"({authors}, {year})"
        
        elif style == 'numeric':
            # Would need reference list to assign number
            return "[1]"  # Placeholder
        
        else:
            return f"[{paper_metadata.get('title', 'Unknown')}]"


# CLI for testing
if __name__ == "__main__":
    print("=== Citation Validator Test ===\n")
    
    validator = CitationValidator()
    auto_citer = AutoCiter(validator)
    
    print("1. Validate citation format")
    print("2. Validate citation context")
    print("3. Check citation in graph")
    print("4. Batch validate text")
    print("5. Suggest citations for text")
    
    choice = input("\nChoice: ")
    
    if choice == "1":
        citation = input("Enter citation: ")
        style = input("Style (apa/mla/chicago): ") or "apa"
        
        result = validator.validate_citation_format(citation, style)
        print("\n" + "="*50)
        print(f"Valid: {result['valid']}")
        if not result['valid']:
            print(f"Error: {result.get('error')}")
            print(f"Suggestion: {result.get('suggestion')}")
    
    elif choice == "2":
        citation = input("Enter citation: ")
        context = input("Enter context: ")
        
        result = validator.validate_citation_context(citation, context)
        print("\n" + "="*50)
        print(result['assessment'])
    
    elif choice == "3":
        citation = input("Enter citation: ")
        
        result = validator.check_citation_exists(citation)
        print("\n" + "="*50)
        print(f"Exists: {result['exists']}")
        if result['exists']:
            print(f"Matches: {result['count']}")
            for match in result.get('matches', []):
                print(f"  - {match}")
    
    elif choice == "4":
        text = input("Enter text with citations: ")
        style = input("Style (apa/mla/chicago): ") or "apa"
        
        result = validator.batch_validate_citations(text, style)
        print("\n" + "="*50)
        print(f"Total citations: {result['total_citations']}")
        print(f"Valid citations: {result['valid_count']}")
        print(f"Accuracy: {result['accuracy']:.1f}%")
        print(f"\nIssues: {len(result['issues'])}")
        for issue in result['issues']:
            print(f"  - {issue}")
    
    elif choice == "5":
        text = input("Enter text to analyze: ")
        
        result = auto_citer.suggest_citations(text)
        print("\n" + "="*50)
        print(result['suggestions'])

