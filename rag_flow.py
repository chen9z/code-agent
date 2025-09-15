import os
import sys
import copy
from typing import Dict, Any, List, Optional
from pathlib import Path

# Import flow/node classes directly from main __init__.py
from __init__ import BaseNode, Node, Flow
from integration.repository import create_repository
from clients.llm import get_default_llm_client
from config.manager import get_config


class RAGIndexNode(Node):
    """Node for indexing a project directory."""
    
    def __init__(self, max_retries: int = 1, wait: int = 0):
        super().__init__(max_retries, wait)
        self.repository = create_repository()
    
    def prep(self, shared: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare for indexing."""
        project_path = self.params.get("project_path") or shared.get("project_path")
        if not project_path:
            raise ValueError("project_path parameter is required")
        
        return {
            "project_path": Path(project_path).expanduser().resolve()
        }
    
    def exec(self, prep_res: Dict[str, Any]) -> Dict[str, Any]:
        """Execute project indexing."""
        project_path = prep_res["project_path"]
        if not project_path.exists():
            raise FileNotFoundError(f"Project directory not found: {project_path}")
        
        if not project_path.is_dir():
            raise ValueError(f"Path is not a directory: {project_path}")
        
        # Index the project
        self.repository.index_project(str(project_path))
        
        # Return the action to determine next node
        action = self.params.get("action") or "index"
        
        return {
            "status": "success",
            "project_name": project_path.name,
            "message": f"Successfully indexed project: {project_path.name}",
            "project_path": str(project_path),
            "action": action  # Return action for flow routing
        }
    
    def exec_fallback(self, prep_res: Dict[str, Any], exc: Exception) -> Dict[str, Any]:
        """Handle indexing failures."""
        return {
            "status": "error",
            "message": f"Indexing failed: {str(exc)}",
            "project_path": str(prep_res["project_path"])
        }


class RAGSearchNode(Node):
    """Node for semantic search in an indexed project."""
    
    def __init__(self, max_retries: int = 1, wait: int = 0):
        super().__init__(max_retries, wait)
        self.repository = create_repository()
    
    def prep(self, shared: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare for search."""
        project_name = self.params.get("project_name") or shared.get("project_name")
        query = self.params.get("query") or shared.get("query")
        limit = self.params.get("limit", 5)
        
        if not project_name:
            raise ValueError("project_name parameter is required")
        if not query:
            raise ValueError("query parameter is required")
        
        return {
            "project_name": project_name,
            "query": query,
            "limit": limit
        }
    
    def exec(self, prep_res: Dict[str, Any]) -> Dict[str, Any]:
        """Execute semantic search."""
        results = self.repository.search(
            prep_res["project_name"], prep_res["query"], prep_res["limit"]
        )
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
            "project_name": prep_res["project_name"],
            "query": prep_res["query"],
            "matches": matches,
            "total_results": len(matches),
            "action": "search",
        }
    
    def exec_fallback(self, prep_res: Dict[str, Any], exc: Exception) -> Dict[str, Any]:
        """Handle search failures."""
        return {
            "status": "error",
            "message": f"Search failed: {str(exc)}",
            "project_name": prep_res["project_name"],
            "query": prep_res["query"],
            "matches": [],
            "total_results": 0
        }


class RAGQueryNode(Node):
    """Node for RAG-based question answering."""
    
    def __init__(self, max_retries: int = 1, wait: int = 0):
        super().__init__(max_retries, wait)
        self.repository = create_repository()
        self.llm = get_default_llm_client()
        self.cfg = get_config()
    
    def prep(self, shared: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare for RAG query."""
        project_name = self.params.get("project_name") or shared.get("project_name")
        # Accept both 'question' and 'query' for compatibility
        question = (
            self.params.get("question")
            or self.params.get("query")
            or shared.get("question")
            or shared.get("query")
        )
        limit = self.params.get("limit", 5)
        
        if not project_name:
            raise ValueError("project_name parameter is required")
        if not question:
            raise ValueError("question parameter is required")
        
        return {
            "project_name": project_name,
            "question": question,
            "limit": limit
        }
    
    def exec(self, prep_res: Dict[str, Any]) -> Dict[str, Any]:
        """Execute RAG query."""
        results = self.repository.search(
            prep_res["project_name"], prep_res["question"], prep_res["limit"]
        )
        if not results:
            return {
                "status": "success",
                "message": "No relevant code context found for your question.",
                "answer": "I couldn't find any relevant code context to answer your question.",
                "action": "query",
            }
        context = ""
        for doc in results:
            context += f"File: {doc.path}\n"
            if hasattr(doc, 'start_line') and hasattr(doc, 'end_line'):
                context += f"Lines: {doc.start_line}-{doc.end_line}\n"
            context += f"Content:\n{doc.content}\n\n"
        messages = [
            {
                "role": "system",
                "content": "You are an expert code assistant. Use the provided code context to answer the user's question accurately and concisely. If the context doesn't contain relevant information, say so.",
            },
            {
                "role": "user",
                "content": f"Code Context:\n{context}\n\nQuestion: {prep_res['question']}\n\nAnswer:",
            },
        ]
        chunks = list(self.llm.get_response(model=self.cfg.llm.model, messages=messages, stream=False))
        answer = "".join(chunks)
        return {
            "status": "success",
            "project_name": prep_res["project_name"],
            "question": prep_res["question"],
            "answer": answer,
            "context_sources": [{
                "file": doc.path,
                "score": getattr(doc, 'score', 0.0)
            } for doc in results],
            "action": "query",
        }
    
    def exec_fallback(self, prep_res: Dict[str, Any], exc: Exception) -> Dict[str, Any]:
        """Handle RAG query failures."""
        return {
            "status": "error",
            "message": f"RAG query failed: {str(exc)}",
            "project_name": prep_res["project_name"],
            "question": prep_res["question"],
            "answer": "Sorry, I couldn't process your question due to an error."
        }


class RAGFlow(Flow):
    """Complete RAG workflow for codebase analysis."""
    
    def __init__(self):
        super().__init__()
        
        # Create nodes
        self.index_node = RAGIndexNode()
        self.search_node = RAGSearchNode()
        self.query_node = RAGQueryNode()
        
        # Set up flow: index -> search/query
        self.start(self.index_node)
        self.index_node.next(self.search_node, "search")
        self.index_node.next(self.query_node, "query")
        
        # Store nodes for dynamic routing
        self.nodes = {
            "index": self.index_node,
            "search": self.search_node,
            "query": self.query_node
        }
    
    def prep(self, shared: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare the RAG flow."""
        # Validate required parameters
        action = self.params.get("action") or shared.get("action")
        if not action:
            raise ValueError("action parameter is required (index, search, or query)")
        
        if action == "index":
            if not self.params.get("project_path") and not shared.get("project_path"):
                raise ValueError("project_path is required for indexing")
        else:
            if not self.params.get("project_name") and not shared.get("project_name"):
                raise ValueError("project_name is required for search/query")
            
            if action == "search" and not self.params.get("query") and not shared.get("query"):
                raise ValueError("query is required for search")
            
            if action == "query" and not (
                self.params.get("question")
                or self.params.get("query")
                or shared.get("question")
                or shared.get("query")
            ):
                raise ValueError("question/query is required for query")
        
        return {"action": action}
    
    def _orch(self, shared: Dict[str, Any], params=None):
        """Run exactly one node based on the requested action and return its result."""
        action = self.params.get("action") or shared.get("action") or "index"
        node = self.nodes.get(action)
        if not node:
            raise ValueError(f"Unknown action: {action}")
        node.set_params(params or {**self.params})
        return node._run(shared)
    
    def post(self, shared: Dict[str, Any], prep_res: Dict[str, Any], exec_res: Any) -> Any:
        """Post-process the RAG flow results."""
        return exec_res


def create_rag_flow() -> RAGFlow:
    """Create a configured RAG flow instance."""
    return RAGFlow()


def run_rag_workflow(action: str, **kwargs) -> Dict[str, Any]:
    """
    Run a complete RAG workflow with a single function call.
    
    Args:
        action: The action to perform (index, search, query)
        **kwargs: Additional parameters for the action
        
    Returns:
        Dictionary with results
    """
    flow = create_rag_flow()
    
    # Set parameters based on action
    params = {"action": action, **kwargs}
    
    # Run the flow
    result = flow.run(params)
    
    return result


# Example usage
if __name__ == "__main__":
    print("Use run_rag_workflow(action=..., **kwargs) from Python.")
