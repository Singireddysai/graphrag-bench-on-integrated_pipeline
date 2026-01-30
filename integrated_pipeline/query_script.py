"""
Query script for LightRAG pipeline.
Queries the RAG system with various questions.
"""
import asyncio
from pathlib import Path
from typing import Optional
from lightrag import LightRAG, QueryParam
from config import Config
from lightrag_utils import initialize_rag
from prompts import get_query_user_prompt
import prompts  # Import to apply custom prompts


async def query_rag(
    rag: LightRAG,
    query: str,
    mode: str = "hybrid",
    chunk_top_k: int = None,
    max_total_tokens: int = None
) -> str:
    """
    Query the RAG system with specified parameters.
    
    Args:
        rag: LightRAG instance
        query: Query string
        mode: Query mode (naive, local, global, hybrid)
        chunk_top_k: Number of top chunks to retrieve
        max_total_tokens: Maximum total tokens for response
        
    Returns:
        Query result string
    """
    # Use defaults from config if not provided
    chunk_top_k = chunk_top_k or Config.DEFAULT_CHUNK_TOP_K
    max_total_tokens = max_total_tokens or Config.DEFAULT_MAX_TOTAL_TOKENS
    
    # Get user prompt to enforce reference format
    user_prompt = get_query_user_prompt()
    
    param = QueryParam(
        mode=mode,
        chunk_top_k=chunk_top_k,
        max_total_tokens=max_total_tokens,
        user_prompt=user_prompt,
    )
    
    result = await rag.aquery(query, param=param)
    return result


async def main(workspace: Optional[str] = None):
    """
    Main function to query the LightRAG system.
    
    Args:
        workspace: Workspace name for data isolation
    """
    rag: LightRAG = None
    
    try:
        # Validate configuration
        Config.validate()
        
        # Check if working directory exists
        working_dir = Path(Config.WORKING_DIR)
        if not working_dir.exists():
            print(f"Error: Working directory '{Config.WORKING_DIR}' not found.")
            print("Please run train_script.py first to index your documents.")
            return
        
        # Initialize RAG instance (loads existing data)
        print("Loading RAG instance (this may take a moment)...")
        if workspace:
            print(f"Using workspace: {workspace}")
        rag = await initialize_rag(workspace=workspace)
        print("✓ RAG instance loaded successfully!\n")
        
        # Define all queries to run
        # queries = [
        #     "What is the value of superficial gas velocity used for simulation of hydrodynamic behaviour of limestone particles?",
        #     "How much efficiency the high performance power systems can have?",
        #     "What is the total height for refractory lined reactor?",
        #     "What is the distance between inlet and outlet of cold model?",
        #     "What is the Sauter mean diameter of plastic particles used in experiments?",
        #     "For the particle size distribution of plastic particles what is the percentage of mass for particles with mean diameter below 1000 micrometer?",
        #     "How many plastic particles have mean diameter below 1000 micrometer expressed in percentage of total mass of al plastic particles?",
        #      "What are the mean particle sizes of excipient powders used in the study to create the powder mixtures?",
        #      "What is the composition of formulation 4?",
        #      "What are the true densities of all powder blends used in study?",
        #      "What are the values of different tooling diameters used for the compactor simulator experiments for differnt campaign across different equipments?",
        #      "What are the MAE values for virtual johnson and johnson models using cs mixed datasets for formulation 2? Which one of them is best model for formulation 2?",
        #      "For Mannitol 200SD, what is the solid fraction value at roll gap 3 mm and roll force of 8 kN/cm, for experiments performed on Gerteis Mini-Pactor®?",
        #      "Compare the solid fraction values from RC experimental datasets for Formulation 1 and Formulation 2 for the first two runs.",
        #      "What are the values of Solid fraction in RC experimental dataset for formulation 3 for roll force of 9 kN/cm and roll gap of 2mm?"
        # ]
        queries = ["What was NVIDIAs total revenue in Q1 Fiscal 2024?",
                    # "By what percentage did Q1 FY2024 revenue increase compared to the previous quarter?",
                    # "What was the GAAP diluted earnings per share for Q1 FY2024?",
                    # "How did NVIDIAs Q1 FY2024 revenue compare year-over-year?",
                    # "Compare GAAP and Non-GAAP diluted EPS for Q1 FY2024.",
                    # "Did operating income grow faster quarter-over-quarter or year-over-year (GAAP)?"
                    # "Which business segment generated the highest revenue in Q1 FY2024?",
                    # "What was the year-over-year growth rate of the Automotive segment in Q1 FY2024?",
                    # "Which business segment experienced the largest year-over-year revenue decline?",
                    # "What is NVIDIA's revenue outlook for Q2 Fiscal 2024?",
                    # "What is the expected Non-GAAP gross margin for Q2 FY2024?",
                    # "What were NVIDIA's net income and net cash provided by operating activities in Q1 FY2024?",
                    # "What was NVIDIA's free cash flow in Q1 FY2024?",
                    # "Which NVIDIA executive spoke about accelerated computing and generative AI?",
                    # "Name two NVIDIA data center products stated to be in production.",
                    # "Did NVIDIA announce any stock buyback program in Q1 FY2024?",
                    # "Was gaming revenue higher than data center revenue in Q1 FY2024?",
                    "What is the abstract for the paper TARGETED FINE-TUNING OF DNN-BASED RECEIVERS VIA INFLUENCE FUNCTIONs ?"]
        # Run all queries in global mode
        print("="*80)
        print("Running all queries in GLOBAL mode")
        print("="*80)
        
        for i, query in enumerate(queries, 1):
            print(f"\n{'='*80}")
            print(f"Query {i}/{len(queries)}")
            print(f"{'='*80}")
            print(f"Query: {query}")
            print(f"Mode: global")
            print(f"{'='*80}\n")
            
            try:
                result = await query_rag(rag, query, mode="global")
                print("\nResult:")
                print(result)
                print("\n" + "-"*80)
            except Exception as e:
                print(f"Error: {e}")
                import traceback
                traceback.print_exc()
                print("\n" + "-"*80)
        
        print("\n" + "="*80)
        print("All queries completed!")
        print("="*80)
        
    except ValueError as e:
        print(f"Configuration Error: {e}")
        print("\nPlease ensure your .env file is properly configured.")
        print("You can copy .env.example to .env and fill in your values.")
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if rag:
            await rag.finalize_storages()
        print("\nDone!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Query LightRAG system",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--workspace", "-w",
        type=str,
        default=None,
        help="Workspace name for data isolation (must match training workspace)"
    )
    
    args = parser.parse_args()
    asyncio.run(main(workspace=args.workspace))

