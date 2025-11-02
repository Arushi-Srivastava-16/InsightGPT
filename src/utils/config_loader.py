"""
Configuration loader for InsightGPT
Supports both .env files and config.ini for backwards compatibility
"""

import os
from pathlib import Path
from configparser import ConfigParser
from typing import Any, Optional
from dotenv import load_dotenv

# Load .env file if it exists
env_path = Path(__file__).parent.parent.parent / "config" / ".env"
if env_path.exists():
    load_dotenv(env_path)
    # Loaded from config/.env
else:
    # Try root .env
    root_env = Path(__file__).parent.parent.parent / ".env"
    if root_env.exists():
        load_dotenv(root_env)
        # Loaded from root .env
    # else: will use config.ini or environment variables


def get_env(key: str, default: Any = None) -> Any:
    """
    Get environment variable with fallback to default
    Supports Streamlit secrets when running in Streamlit
    
    Args:
        key: Environment variable name
        default: Default value if not found
        
    Returns:
        Environment variable value or default
    """
    # First try environment variables
    value = os.getenv(key)
    if value is not None:
        return value
    
    # Try Streamlit secrets if available
    try:
        import streamlit as st
        if hasattr(st, 'secrets') and key in st.secrets:
            return st.secrets[key]
    except (ImportError, AttributeError, KeyError):
        pass
    
    return default


def get_env_bool(key: str, default: bool = False) -> bool:
    """
    Get boolean environment variable
    
    Args:
        key: Environment variable name
        default: Default boolean value
        
    Returns:
        Boolean value
    """
    value = os.getenv(key, str(default)).lower()
    return value in ('true', '1', 'yes', 'on')


def get_env_int(key: str, default: int = 0) -> int:
    """
    Get integer environment variable
    
    Args:
        key: Environment variable name
        default: Default integer value
        
    Returns:
        Integer value
    """
    try:
        return int(os.getenv(key, default))
    except (ValueError, TypeError):
        return default


def get_env_float(key: str, default: float = 0.0) -> float:
    """
    Get float environment variable
    
    Args:
        key: Environment variable name
        default: Default float value
        
    Returns:
        Float value
    """
    try:
        return float(os.getenv(key, default))
    except (ValueError, TypeError):
        return default


def load_config(use_env: bool = True) -> ConfigParser:
    """
    Load configuration with priority: .env > config.ini > defaults
    
    Args:
        use_env: Whether to prioritize environment variables
        
    Returns:
        ConfigParser object with merged configuration
    """
    config = ConfigParser()
    
    # Try to load config.ini
    config_paths = [
        Path(__file__).parent.parent.parent / "config" / "config.ini",
        Path(__file__).parent.parent.parent / "config.ini",
    ]
    
    for config_path in config_paths:
        if config_path.exists():
            config.read(config_path)
            # Loaded config from file
            break
    
    # If no config.ini, create default structure
    if not config.sections():
        config.add_section('Neo4j')
        config.add_section('LLM')
        config.add_section('OpenAI')
        config.add_section('Ollama')
        config.add_section('Embeddings')
        config.add_section('Chat')
        config.add_section('Zotero')
        config.add_section('PDF')
    
    # Override with environment variables if requested
    if use_env:
        # Neo4j
        if 'Neo4j' in config:
            config['Neo4j']['uri'] = get_env('NEO4J_URI', config.get('Neo4j', 'uri', fallback='bolt://localhost:7687'))
            config['Neo4j']['username'] = get_env('NEO4J_USERNAME', config.get('Neo4j', 'username', fallback='neo4j'))
            config['Neo4j']['password'] = get_env('NEO4J_PASSWORD', config.get('Neo4j', 'password', fallback=''))
        
        # LLM
        if 'LLM' in config:
            llm_provider = get_env('LLM_PROVIDER', config.get('LLM', 'llm', fallback='Ollama'))
            config['LLM']['llm'] = llm_provider.capitalize()
            config['LLM']['temperature'] = str(get_env_float('LLM_TEMPERATURE', 
                                                              float(config.get('LLM', 'temperature', fallback='0.0'))))
            config['LLM']['max_tokens'] = str(get_env_int('LLM_MAX_TOKENS',
                                                           int(config.get('LLM', 'max_tokens', fallback='2048'))))
            config['LLM']['stop'] = config.get('LLM', 'stop', fallback='["<|im_end|>"]')
        
        # OpenAI
        if 'OpenAI' in config:
            config['OpenAI']['api_key'] = get_env('OPENAI_API_KEY', config.get('OpenAI', 'api_key', fallback=''))
            config['OpenAI']['model'] = get_env('OPENAI_MODEL', config.get('OpenAI', 'model', fallback='gpt-3.5-turbo'))
            api_base = get_env('OPENAI_API_BASE')
            if api_base:
                config['OpenAI']['api_base'] = api_base
        
        # Ollama
        if 'Ollama' in config:
            config['Ollama']['model'] = get_env('OLLAMA_MODEL', 
                                                 config.get('Ollama', 'model', fallback='hermes-2-pro-llama-3-8b'))
            config['Ollama']['num_ctx'] = str(get_env_int('OLLAMA_NUM_CTX',
                                                           int(config.get('Ollama', 'num_ctx', fallback='2048'))))
        
        # Embeddings
        if 'Embeddings' in config:
            embeddings_provider = get_env('EMBEDDINGS_PROVIDER', config.get('Embeddings', 'embeddings', fallback='Ollama'))
            config['Embeddings']['embeddings'] = embeddings_provider.capitalize()
            
            if embeddings_provider.lower() == 'openai':
                config['Embeddings']['model'] = get_env('OPENAI_EMBEDDING_MODEL',
                                                         config.get('Embeddings', 'model', fallback='text-embedding-3-small'))
            else:
                config['Embeddings']['model'] = get_env('OLLAMA_EMBEDDING_MODEL',
                                                         config.get('Embeddings', 'model', fallback='mxbai-embed-large'))
        
        # Chat
        if 'Chat' in config:
            chat_provider = get_env('CHAT_PROVIDER', config.get('Chat', 'chatbot', fallback='Ollama'))
            config['Chat']['chatbot'] = chat_provider.capitalize()
            config['Chat']['model'] = get_env('CHAT_MODEL', config.get('Chat', 'model', fallback='hermes-2-pro-llama-3-8b'))
            config['Chat']['temperature'] = str(get_env_float('CHAT_TEMPERATURE',
                                                               float(config.get('Chat', 'temperature', fallback='0.8'))))
            config['Chat']['max_tokens'] = str(get_env_int('CHAT_MAX_TOKENS',
                                                            int(config.get('Chat', 'max_tokens', fallback='2048'))))
            config['Chat']['stop'] = config.get('Chat', 'stop', fallback='["<|im_end|>","USER:"]')
        
        # Zotero
        if 'Zotero' in config:
            config['Zotero']['enabled'] = str(get_env_bool('ZOTERO_ENABLED',
                                                            config.getboolean('Zotero', 'enabled', fallback=False)))
            config['Zotero']['library_id'] = get_env('ZOTERO_LIBRARY_ID',
                                                      config.get('Zotero', 'library_id', fallback=''))
            config['Zotero']['library_type'] = get_env('ZOTERO_LIBRARY_TYPE',
                                                        config.get('Zotero', 'library_type', fallback='user'))
            config['Zotero']['api_key'] = get_env('ZOTERO_API_KEY',
                                                   config.get('Zotero', 'api_key', fallback=''))
            storage_dir = get_env('ZOTERO_STORAGE_DIR')
            if storage_dir:
                config['Zotero']['Zotero_dir'] = storage_dir
        
        # PDF
        if 'PDF' in config:
            config['PDF']['extract_images'] = str(get_env_bool('PDF_EXTRACT_IMAGES',
                                                                config.getboolean('PDF', 'extract_images', fallback=False)))
            config['PDF']['max_char'] = str(get_env_int('PDF_MAX_CHAR',
                                                         int(config.get('PDF', 'max_char', fallback='1000'))))
            config['PDF']['new_after_n_chars'] = str(get_env_int('PDF_NEW_AFTER_N_CHARS',
                                                                  int(config.get('PDF', 'new_after_n_chars', fallback='800'))))
            config['PDF']['combine_text_under_n_chars'] = str(get_env_int('PDF_COMBINE_TEXT_UNDER_N_CHARS',
                                                                           int(config.get('PDF', 'combine_text_under_n_chars', fallback='200'))))
    
    return config


def get_neo4j_config() -> dict:
    """Get Neo4j configuration"""
    return {
        'uri': get_env('NEO4J_URI', 'bolt://localhost:7687'),
        'username': get_env('NEO4J_USERNAME', 'neo4j'),
        'password': get_env('NEO4J_PASSWORD', ''),
    }


def get_llm_config() -> dict:
    """Get LLM configuration"""
    provider = get_env('LLM_PROVIDER', 'ollama').lower()
    
    config = {
        'provider': provider,
        'temperature': get_env_float('LLM_TEMPERATURE', 0.0),
        'max_tokens': get_env_int('LLM_MAX_TOKENS', 2048),
    }
    
    if provider == 'openai':
        config['api_key'] = get_env('OPENAI_API_KEY', '')
        config['model'] = get_env('OPENAI_MODEL', 'gpt-3.5-turbo')
        api_base = get_env('OPENAI_API_BASE')
        if api_base:
            config['api_base'] = api_base
    else:
        config['model'] = get_env('OLLAMA_MODEL', 'hermes-2-pro-llama-3-8b')
        config['num_ctx'] = get_env_int('OLLAMA_NUM_CTX', 2048)
    
    return config


def validate_config() -> tuple[bool, list[str]]:
    """
    Validate that required configuration is present
    
    Returns:
        Tuple of (is_valid, list of missing items)
    """
    missing = []
    
    # Check Neo4j
    if not get_env('NEO4J_PASSWORD'):
        missing.append('NEO4J_PASSWORD')
    
    # Check LLM provider
    provider = get_env('LLM_PROVIDER', 'ollama').lower()
    if provider == 'openai':
        if not get_env('OPENAI_API_KEY'):
            missing.append('OPENAI_API_KEY')
    
    return (len(missing) == 0, missing)


# Auto-validate on import
if __name__ != "__main__":
    is_valid, missing = validate_config()
    # Validation happens silently at import time
    # Errors will be caught by the application if needed


