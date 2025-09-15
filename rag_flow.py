import os
import sys
import copy
from typing import Dict, Any, List, Optional
from pathlib import Path

# Import flow/node classes directly from main __init__.py
from __init__ import BaseNode, Node, Flow
from tools.rag_tool import create_rag_tool


class RAGIndexNode(Node):
    """Node for indexing a project directory."""
    
    def __init__(self, max_retries: int = 1, wait: int = 0):
        super().__init__(max_retries, wait)
        self.rag_tool = None
    
    def prep(self, shared: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare for indexing."""
        if self.rag_tool is None:
            self.rag_tool = create_rag_tool()
        
        project_path = self.params.get("project_path") or shared.get("project_path")
        if not project_path:
            raise ValueError("project_path parameter is required")
        
        return {
            "project_path": Path(project_path).expanduser().resolve(),
            "rag_tool": self.rag_tool
        }
    
    def exec(self, prep_res: Dict[str, Any]) -> Dict[str, Any]:
        """Execute project indexing."""
        project_path = prep_res["project_path"]
        rag_tool = prep_res["rag_tool"]
        
        if not project_path.exists():
            raise FileNotFoundError(f"Project directory not found: {project_path}")
        
        if not project_path.is_dir():
            raise ValueError(f"Path is not a directory: {project_path}")
        
        # Index the project
        result = rag_tool.execute(
            action="index",
            project_path=str(project_path)
        )
        
        if result["status"] != "success":
            raise RuntimeError(f"Indexing failed: {result.get('message', 'Unknown error')}")
        
        # Return the action to determine next node
        action = self.params.get("action") or "index"
        
        return {
            "status": "success",
            "project_name": result.get("project_name", project_path.name),
            "message": result.get("message", "Project indexed successfully"),
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
        self.rag_tool = None
    
    def prep(self, shared: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare for search."""
        if self.rag_tool is None:
            self.rag_tool = create_rag_tool()
        
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
            "limit": limit,
            "rag_tool": self.rag_tool
        }
    
    def exec(self, prep_res: Dict[str, Any]) -> Dict[str, Any]:
        """Execute semantic search."""
        rag_tool = prep_res["rag_tool"]
        
        result = rag_tool.execute(
            action="search",
            project_name=prep_res["project_name"],
            query=prep_res["query"],
            limit=prep_res["limit"]
        )
        
        if result["status"] != "success":
            raise RuntimeError(f"Search failed: {result.get('message', 'Unknown error')}")
        
        # Return action for flow routing
        result["action"] = "search"
        return result
    
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
        self.rag_tool = None
    
    def prep(self, shared: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare for RAG query."""
        if self.rag_tool is None:
            self.rag_tool = create_rag_tool()
        
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
            "limit": limit,
            "rag_tool": self.rag_tool
        }
    
    def exec(self, prep_res: Dict[str, Any]) -> Dict[str, Any]:
        """Execute RAG query."""
        rag_tool = prep_res["rag_tool"]
        
        result = rag_tool.execute(
            action="query",
            project_name=prep_res["project_name"],
            # 'execute' accepts aliasing; pass as 'question' for clarity
            question=prep_res["question"],
            limit=prep_res["limit"]
        )
        
        if result["status"] != "success":
            raise RuntimeError(f"RAG query failed: {result.get('message', 'Unknown error')}")
        
        # Return action for flow routing
        result["action"] = "query"
        return result
    
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
        """Custom orchestration to handle different starting points based on action."""
        action = self.params.get("action") or shared.get("action") or "index"
        
        # Start with the appropriate node based on action
        curr = copy.copy(self.nodes.get(action, self.index_node))
        p = (params or {**self.params})
        last_action = None
        
        while curr:
            curr.set_params(p)
            last_action = curr._run(shared)
            curr = copy.copy(self.get_next_node(curr, last_action))
        
        return last_action
    
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
    # Example 1: Index a project
    print("Example 1: Indexing a project")
    try:
        result = run_rag_workflow(
            action="index",
            project_path="/tmp/test_project"
        )
        print(f"Result: {result}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Example 2: Search for code
    print("\nExample 2: Searching for code")
    try:
        result = run_rag_workflow(
            action="search",
            project_name="test_project",
            query="function definition"
        )
        print(f"Found {result.get('total_results', 0)} results")
    except Exception as e:
        print(f"Error: {e}")
    
    # Example 3: Ask a question
    print("\nExample 3: Asking a question")
    try:
        result = run_rag_workflow(
            action="query",
            project_name="test_project",
            question="What does this code do?"
        )
        print(f"Answer: {result.get('answer', 'No answer')}")
    except Exception as e:
        print(f"Error: {e}")
