import os
from typing import Any, Dict, Generator, Optional
from pathlib import Path

from tools.base import BaseTool
from config.manager import get_config
from integration.repository import RepositoryAdapter, create_repository
from clients.llm import get_default_llm_client


"""LLM client selection is handled by clients.llm.get_default_llm_client."""


class RAGTool(BaseTool):
    """Tool for codebase retrieval augmented generation."""

    def __init__(self, 
                 repository: Optional[RepositoryAdapter] = None,
                 llm_client=None):
        """
        Initialize the RAG tool.
        
        Args:
            repository: Repository adapter instance
            llm_client: LLM client instance
        """
        self.config = get_config()
        self.repository = repository or create_repository()
        self.llm_client = llm_client or get_default_llm_client()
    
    @property
    def name(self) -> str:
        return "codebase_rag"

    @property
    def description(self) -> str:
        return """Perform semantic search and retrieval augmented generation on codebases. 
        This tool can index projects, search for relevant code, and answer questions 
        using code context from the indexed projects."""

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["index", "search", "query"],
                    "description": "Action to perform: index a project, search code, or ask a question"
                },
                "project_path": {
                    "type": "string",
                    "description": "Path to project directory (for indexing)"
                },
                "project_name": {
                    "type": "string", 
                    "description": "Name of the project to search/query (use basename of path)"
                },
                "query": {
                    "type": "string",
                    "description": "Search query or question to ask"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of search results to return",
                    "default": 5
                }
            },
            "required": ["action"],
            "additionalProperties": False
        }

    def execute(self, **kwargs) -> Any:
        """
        Execute the RAG tool.
        
        Supported actions:
        - index: Index a project directory
        - search: Perform semantic search on indexed project
        - query: Ask a question using RAG
        """
        action = kwargs.get("action")
        
        # Alias normalization
        if action == "query" and "query" not in kwargs and "question" in kwargs:
            kwargs["query"] = kwargs.get("question")

        if action == "index":
            return self._index_project(**kwargs)
        elif action == "search":
            return self._semantic_search(**kwargs)
        elif action == "query":
            return self._rag_query(**kwargs)
        else:
            raise ValueError(f"Unknown action: {action}")

    def _index_project(self, project_path: str, **kwargs) -> Dict[str, Any]:
        """Index a project directory."""
        try:
            self.repository.index_project(project_path)
            project_name = Path(project_path).name
            return {
                "status": "success",
                "message": f"Successfully indexed project: {project_name}",
                "project_name": project_name
            }
        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to index project: {str(e)}"
            }

    def _semantic_search(self, project_name: str, query: str, limit: int = 5, **kwargs) -> Dict[str, Any]:
        """Perform semantic search on indexed project."""
        try:
            results = self.repository.search(project_name, query, limit)
            
            # Format results
            matches = []
            for doc in results:
                matches.append({
                    "file": doc.path,
                    "chunk_id": getattr(doc, 'chunk_id', None),
                    "content": doc.content,
                    "score": getattr(doc, 'score', 0.0),
                    "start_line": getattr(doc, 'start_line', 0),
                    "end_line": getattr(doc, 'end_line', 0)
                })
            
            return {
                "status": "success",
                "project_name": project_name,
                "query": query,
                "matches": matches,
                "total_results": len(matches)
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Search failed: {str(e)}",
                "matches": []
            }

    def _rag_query(self, project_name: str, query: str, limit: int = 5, **kwargs) -> Dict[str, Any]:
        """Perform RAG-based question answering."""
        try:
            # First perform semantic search to get relevant context
            results = self.repository.search(project_name, query, limit)
            
            if not results:
                return {
                    "status": "success",
                    "message": "No relevant code context found for your question.",
                    "answer": "I couldn't find any relevant code context to answer your question. "
                             "Please try a different query or ensure the project is properly indexed."
                }
            
            # Format context from search results
            context = ""
            for doc in results:
                context += f"File: {doc.path}\n"
                if hasattr(doc, 'start_line') and hasattr(doc, 'end_line'):
                    context += f"Lines: {doc.start_line}-{doc.end_line}\n"
                context += f"Content:\n{doc.content}\n\n"
            
            # Create prompt for LLM
            messages = [
                {
                    "role": "system",
                    "content": "You are an expert code assistant. Use the provided code context "
                             "to answer the user's question accurately and concisely. "
                             "If the context doesn't contain relevant information, say so."
                },
                {
                    "role": "user",
                    "content": f"Code Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
                }
            ]
            
            # Get response from LLM
            response_gen = self.llm_client.get_response(
                model=self.config.llm.model,
                messages=messages,
                stream=False
            )
            
            # Collect the response (since it's a generator)
            answer = "".join(list(response_gen))
            
            return {
                "status": "success",
                "project_name": project_name,
                "question": query,
                "answer": answer,
                "context_sources": [{
                    "file": doc.path,
                    "score": getattr(doc, 'score', 0.0)
                } for doc in results]
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"RAG query failed: {str(e)}"
            }

    def get_streaming_response(self, project_name: str, query: str, limit: int = 5) -> Generator:
        """Get streaming response for RAG query."""
        try:
            # Perform semantic search
            results = self.repository.search(project_name, query, limit)
            
            if not results:
                yield "No relevant code context found for your question."
                return
            
            # Format context
            context = ""
            for doc in results:
                context += f"File: {doc.path}\n"
                if hasattr(doc, 'start_line') and hasattr(doc, 'end_line'):
                    context += f"Lines: {doc.start_line}-{doc.end_line}\n"
                context += f"Content:\n{doc.content}\n\n"
            
            # Create prompt
            messages = [
                {
                    "role": "system",
                    "content": "You are an expert code assistant. Use the provided code context "
                             "to answer the user's question accurately and concisely."
                },
                {
                    "role": "user",
                    "content": f"Code Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
                }
            ]
            
            # Stream response
            for chunk in self.llm_client.get_response(
                model=self.config.llm.model,
                messages=messages,
                stream=True
            ):
                yield chunk
                
        except Exception as e:
            yield f"Error: {str(e)}"


def create_rag_tool() -> RAGTool:
    """Create a configured RAG tool instance."""
    return RAGTool()
