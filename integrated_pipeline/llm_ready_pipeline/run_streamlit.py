#!/usr/bin/env python3
"""
Launcher script for the LLM-Ready PDF Extraction Streamlit UI
"""

import subprocess
import sys
from pathlib import Path

def main():
    """Launch the Streamlit app."""
    script_dir = Path(__file__).parent
    app_path = script_dir / "streamlit_app.py"
    
    if not app_path.exists():
        print(f"❌ Error: streamlit_app.py not found at {app_path}")
        sys.exit(1)
    
    try:
        print("🚀 Starting LLM-Ready PDF Extraction Streamlit UI...")
        print("📄 The app will open in your default web browser")
        print("🔗 URL: http://localhost:8501")
        print("\n💡 Press Ctrl+C to stop the server")
        print("=" * 60)
        
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            str(app_path),
            "--server.port", "8501",
            "--server.address", "localhost",
            "--browser.gatherUsageStats", "false"
        ])
        
    except KeyboardInterrupt:
        print("\n👋 Streamlit app stopped.")
    except Exception as e:
        print(f"❌ Error launching Streamlit: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()