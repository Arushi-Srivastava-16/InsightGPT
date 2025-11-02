"""
Fallback PDF processing for Streamlit Cloud
Uses only basic libraries that work in cloud environments
"""

import os
from typing import List
from langchain.schema import Document
import tempfile

def simple_pdf_extract(file_path: str, metadata: dict, max_char: int = 1000) -> List[Document]:
    """
    Simple PDF text extraction using multiple fallback methods
    """
    documents = []
    
    # Method 1: Try PyMuPDF (fitz)
    try:
        import fitz  # PyMuPDF
        print("INFO: Using PyMuPDF for PDF extraction")
        
        doc = fitz.open(file_path)
        full_text = ""
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text()
            full_text += f"\n--- Page {page_num + 1} ---\n{text}"
        
        doc.close()
        
        # Chunk the text
        documents = chunk_text(full_text, metadata, max_char)
        print(f"SUCCESS: Extracted {len(documents)} chunks using PyMuPDF")
        return documents
        
    except ImportError:
        print("WARNING: PyMuPDF not available, trying pypdf...")
    except Exception as e:
        print(f"WARNING: PyMuPDF failed: {e}, trying pypdf...")
    
    # Method 2: Try pypdf
    try:
        from pypdf import PdfReader
        print("INFO: Using pypdf for PDF extraction")
        
        reader = PdfReader(file_path)
        full_text = ""
        
        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()
            full_text += f"\n--- Page {page_num + 1} ---\n{text}"
        
        # Chunk the text
        documents = chunk_text(full_text, metadata, max_char)
        print(f"SUCCESS: Extracted {len(documents)} chunks using pypdf")
        return documents
        
    except ImportError:
        print("WARNING: pypdf not available, trying pdfminer...")
    except Exception as e:
        print(f"WARNING: pypdf failed: {e}, trying pdfminer...")
    
    # Method 3: Try pdfminer
    try:
        from pdfminer.high_level import extract_text
        print("INFO: Using pdfminer for PDF extraction")
        
        full_text = extract_text(file_path)
        
        # Chunk the text
        documents = chunk_text(full_text, metadata, max_char)
        print(f"SUCCESS: Extracted {len(documents)} chunks using pdfminer")
        return documents
        
    except ImportError:
        print("ERROR: No PDF libraries available!")
        raise ImportError("No PDF processing libraries available. Please install pymupdf, pypdf, or pdfminer.")
    except Exception as e:
        print(f"ERROR: All PDF extraction methods failed: {e}")
        raise e

def chunk_text(text: str, metadata: dict, max_char: int = 1000) -> List[Document]:
    """
    Simple text chunking
    """
    documents = []
    
    # Clean the text
    text = text.strip()
    if not text:
        return documents
    
    # Simple chunking by character count
    chunks = []
    current_chunk = ""
    
    # Split by paragraphs first
    paragraphs = text.split('\n\n')
    
    for paragraph in paragraphs:
        if len(current_chunk) + len(paragraph) <= max_char:
            current_chunk += paragraph + "\n\n"
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = paragraph + "\n\n"
    
    # Add the last chunk
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    # Create Document objects
    for i, chunk in enumerate(chunks):
        if chunk.strip():  # Only add non-empty chunks
            doc_metadata = metadata.copy()
            doc_metadata.update({
                'chunk_id': i,
                'total_chunks': len(chunks),
                'chunk_size': len(chunk)
            })
            
            documents.append(Document(
                page_content=chunk,
                metadata=doc_metadata
            ))
    
    return documents

def process_pdf_simple(file_path: str, metadata: dict, max_char: int = 1000) -> List[Document]:
    """
    Main function for simple PDF processing
    """
    try:
        return simple_pdf_extract(file_path, metadata, max_char)
    except Exception as e:
        print(f"ERROR: Simple PDF processing failed: {e}")
        # Return a single document with error message
        return [Document(
            page_content=f"Error processing PDF: {str(e)}",
            metadata={**metadata, 'error': True}
        )]
