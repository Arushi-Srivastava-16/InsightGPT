"""
InsightGPT - AI-Powered Research Copilot
Main Entry Point for Streamlit Web Application

Run with: streamlit run app.py
"""

import sys
from pathlib import Path

# Add src to path so 'src' is importable
sys.path.insert(0, str(Path(__file__).parent))

# Import the working single-page UI
import src.ui.app_single_page
