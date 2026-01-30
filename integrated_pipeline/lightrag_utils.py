"""
LightRAG utility functions for initialization and setup.
"""
import os
from typing import Optional
from lightrag import LightRAG
from lightrag.llm.openai import openai_embed
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.utils import EmbeddingFunc
from config import Config


async def create_embedding_func() -> EmbeddingFunc:
    """
    Create and configure the embedding function.
    
    Returns:
        Configured EmbeddingFunc instance
    """
    embedding_func = EmbeddingFunc(
        embedding_dim=Config.EMBEDDING_DIM,
        func=lambda texts: openai_embed(
            texts,
            model=Config.EMBEDDING_MODEL,
            base_url=Config.OPENAI_API_BASE,
            api_key=Config.OPENAI_API_KEY
        )
    )
    return embedding_func


async def create_llm_func():
    """
    Create and configure the LLM function for OpenRouter.
    
    Returns:
        Async function for LLM completion
    """
    async def openrouter_complete(prompt, system_prompt=None, history_messages=[], **kwargs):
        from lightrag.llm.openai import openai_complete_if_cache
        
        # Remove model from kwargs if present (we set it explicitly)
        if "model" in kwargs:
            del kwargs["model"]
        
        # Set default max_tokens for training
        kwargs.setdefault("max_tokens", Config.LLM_MAX_TOKENS)
        
        return await openai_complete_if_cache(
            Config.LLM_MODEL,
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            **kwargs
        )
    
    return openrouter_complete


async def initialize_rag(
    working_dir: Optional[str] = None,
    workspace: Optional[str] = None,
    llm_model: Optional[str] = None,
    llm_max_tokens: Optional[int] = None,
    chunk_token_size: Optional[int] = None,
    chunk_overlap_token_size: Optional[int] = None,
    enable_llm_cache: Optional[bool] = None,
    enable_llm_cache_for_entity_extract: Optional[bool] = None,
) -> LightRAG:
    """
    Initialize LightRAG instance with configuration.
    
    Args:
        working_dir: Working directory for RAG (defaults to Config.WORKING_DIR)
        workspace: Workspace name for data isolation (defaults to Config.WORKSPACE)
        llm_model: LLM model to use (defaults to Config.LLM_MODEL)
        llm_max_tokens: Max tokens for LLM (defaults to Config.LLM_MAX_TOKENS)
        chunk_token_size: Token size for chunks (defaults to Config.CHUNK_TOKEN_SIZE)
        chunk_overlap_token_size: Overlap size for chunks (defaults to Config.CHUNK_OVERLAP_TOKEN_SIZE)
        enable_llm_cache: Enable LLM caching (defaults to Config.ENABLE_LLM_CACHE)
        enable_llm_cache_for_entity_extract: Enable caching for entity extraction (defaults to Config.ENABLE_LLM_CACHE_FOR_ENTITY_EXTRACT)
        
    Returns:
        Initialized LightRAG instance
    """
    # Validate configuration
    Config.validate()
    
    # Setup environment
    Config.setup_environment()
    
    # Use provided values or fall back to config
    working_dir = working_dir or Config.WORKING_DIR
    workspace = workspace or Config.WORKSPACE
    chunk_token_size = chunk_token_size or Config.CHUNK_TOKEN_SIZE
    chunk_overlap_token_size = chunk_overlap_token_size or Config.CHUNK_OVERLAP_TOKEN_SIZE
    enable_llm_cache = enable_llm_cache if enable_llm_cache is not None else Config.ENABLE_LLM_CACHE
    enable_llm_cache_for_entity_extract = (
        enable_llm_cache_for_entity_extract 
        if enable_llm_cache_for_entity_extract is not None 
        else Config.ENABLE_LLM_CACHE_FOR_ENTITY_EXTRACT
    )
    
    # Store original values for restoration
    original_model = Config.LLM_MODEL
    original_max_tokens = Config.LLM_MAX_TOKENS
    
    # Temporarily override model if provided
    if llm_model:
        Config.LLM_MODEL = llm_model
    if llm_max_tokens:
        Config.LLM_MAX_TOKENS = llm_max_tokens
    
    try:
        # Create embedding function
        embedding_func = await create_embedding_func()
        
        # Create LLM function
        llm_func = await create_llm_func()
        
        # Initialize LightRAG
        rag = LightRAG(
            working_dir=working_dir,
            workspace=workspace,
            embedding_func=embedding_func,
            llm_model_func=llm_func,
            graph_storage=Config.GRAPH_STORAGE,
            vector_storage=Config.VECTOR_STORAGE,
            chunk_token_size=chunk_token_size,
            chunk_overlap_token_size=chunk_overlap_token_size,
            cosine_threshold=Config.COSINE_THRESHOLD,
            enable_llm_cache=enable_llm_cache,
            enable_llm_cache_for_entity_extract=enable_llm_cache_for_entity_extract,
        )
        
        # Initialize storages
        await rag.initialize_storages()
        await initialize_pipeline_status()
        
        return rag
    finally:
        # Restore original values
        if llm_model:
            Config.LLM_MODEL = original_model
        if llm_max_tokens:
            Config.LLM_MAX_TOKENS = original_max_tokens


async def test_embedding_function(rag: LightRAG) -> None:
    """
    Test the embedding function to verify it's working correctly.
    
    Args:
        rag: LightRAG instance
    """
    print("\nTesting embedding function...")
    test_text = ["This is a test string for embedding."]
    embedding = await rag.embedding_func(test_text)
    embedding_dim = embedding.shape[1]
    print(f"[OK] Embedding dimension: {embedding_dim}")

