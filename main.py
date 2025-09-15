#!/usr/bin/env python3
"""
Main entry point for the code-agent with RAG integration.
"""

import os
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def main():
    """Main function demonstrating RAG integration."""
    print("🚀 Code Agent with RAG Integration")
    print("=" * 50)
    
    # Check if chat-codebase is available via env var
    chat_codebase_path = os.getenv("CHAT_CODEBASE_PATH")
    if not chat_codebase_path or not Path(chat_codebase_path).exists():
        print("⚠️  chat-codebase not detected.")
        print("   Set CHAT_CODEBASE_PATH to the repo root if available.")
        print("   Example: export CHAT_CODEBASE_PATH=~/workspace/chat-codebase")
        print("\nPath reference (update if needed):")
        print("   integration/repository.py")
        print("\nRunning in fallback mode (stub repository)...")
    
    # Check environment variables
    if not os.getenv("OPENAI_API_BASE") or not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Environment variables not set:")
        print("   Please set OPENAI_API_BASE and OPENAI_API_KEY")
        print("\nExample:")
        print("   export OPENAI_API_BASE=https://api.deepseek.com")
        print("   export OPENAI_API_KEY=your_api_key_here")
        print("\nContinuing with limited functionality...")
    
    # Demonstrate RAG tool creation
    try:
        from tools.rag_tool import create_rag_tool
        
        rag_tool = create_rag_tool()
        print(f"✅ RAG Tool created successfully: {rag_tool.name}")
        print(f"   Description: {rag_tool.description}")
        
        # Show tool parameters
        params = rag_tool.parameters
        print(f"\n🔧 Available actions: {params['properties']['action']['enum']}")
        
        print("\n📋 Usage Examples:")
        print("1. Index a project:")
        print("   rag_tool.execute(action='index', project_path='/path/to/project')")
        
        print("\n2. Search for code:")
        print("   rag_tool.execute(action='search', project_name='project_name', query='function definition')")
        
        print("\n3. Ask a question:")
        print("   rag_tool.execute(action='query', project_name='project_name', query='How does authentication work?')")
        
    except ImportError as e:
        print(f"❌ Failed to import RAG tool: {e}")
        print("\n💡 Make sure all dependencies are installed:")
        print("   uv pip install -e '.[test]'")
    except Exception as e:
        print(f"❌ Error creating RAG tool: {e}")
    
    print("\n" + "=" * 50)
    print("🎯 Ready to use codebase RAG functionality!")

if __name__ == "__main__":
    main()
