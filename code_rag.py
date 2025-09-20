from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from __init__ import Flow, Node
from integrations.repository import create_repository
from integrations.tree_sitter import TreeSitterProjectParser
from clients.llm import get_default_llm_client
from configs.manager import get_config


class RAGIndexNode(Node):
    """Node for indexing a project directory."""

    def __init__(self, max_retries: int = 1, wait: int = 0) -> None:
        super().__init__(max_retries, wait)
        self.repository = create_repository()
        self.ts_parser = TreeSitterProjectParser()

    def prep(self, shared: Dict[str, Any]) -> Dict[str, Any]:
        project_path = self.params.get("project_path") or shared.get("project_path")
        if not project_path:
            raise ValueError("project_path parameter is required")
        return {"project_path": Path(project_path).expanduser().resolve()}

    def exec(self, prep_res: Dict[str, Any]) -> Dict[str, Any]:
        project_path: Path = prep_res["project_path"]
        if not project_path.exists():
            raise FileNotFoundError(f"Project directory not found: {project_path}")
        if not project_path.is_dir():
            raise ValueError(f"Path is not a directory: {project_path}")

        self.repository.index_project(str(project_path))

        symbols_error: Optional[str] = None
        try:
            symbols = self.ts_parser.parse_project(project_path)
            storage_dir = Path("storage/tree_sitter").resolve()
            storage_dir.mkdir(parents=True, exist_ok=True)
            symbols_path = storage_dir / f"{project_path.name}_symbols.jsonl"
            self.ts_parser.export_symbols(symbols, symbols_path)
            symbols_path_str: Optional[str] = str(symbols_path)
        except Exception as exc:
            symbols = []
            symbols_path_str = None
            symbols_error = str(exc)
        action = self.params.get("action") or "index"

        return {
            "status": "success",
            "project_name": project_path.name,
            "message": f"Successfully indexed project: {project_path.name}",
            "project_path": str(project_path),
            "parsed_symbols_count": len(symbols),
            "parsed_symbols_path": symbols_path_str,
            "parsed_symbols_error": symbols_error,
            "action": action,
        }

    def exec_fallback(self, prep_res: Dict[str, Any], exc: Exception) -> Dict[str, Any]:
        return {
            "status": "error",
            "message": f"Indexing failed: {exc}",
            "project_path": str(prep_res["project_path"]),
        }


class RAGSearchNode(Node):
    """Node for semantic search in an indexed project."""

    def __init__(self, max_retries: int = 1, wait: int = 0) -> None:
        super().__init__(max_retries, wait)
        self.repository = create_repository()

    def prep(self, shared: Dict[str, Any]) -> Dict[str, Any]:
        project_name = self.params.get("project_name") or shared.get("project_name")
        query = self.params.get("query") or shared.get("query")
        limit = self.params.get("limit", 5)
        if not project_name:
            raise ValueError("project_name parameter is required")
        if not query:
            raise ValueError("query parameter is required")
        return {"project_name": project_name, "query": query, "limit": limit}

    def exec(self, prep_res: Dict[str, Any]) -> Dict[str, Any]:
        results = self.repository.search(
            prep_res["project_name"], prep_res["query"], prep_res["limit"]
        )
        matches: List[Dict[str, Any]] = []
        for doc in results:
            matches.append(
                {
                    "file": doc.path,
                    "chunk_id": getattr(doc, "chunk_id", None),
                    "content": doc.content,
                    "score": getattr(doc, "score", 0.0),
                    "start_line": getattr(doc, "start_line", 0),
                    "end_line": getattr(doc, "end_line", 0),
                }
            )
        return {
            "status": "success",
            "project_name": prep_res["project_name"],
            "query": prep_res["query"],
            "matches": matches,
            "total_results": len(matches),
            "action": "search",
        }

    def exec_fallback(self, prep_res: Dict[str, Any], exc: Exception) -> Dict[str, Any]:
        return {
            "status": "error",
            "message": f"Search failed: {exc}",
            "project_name": prep_res["project_name"],
            "query": prep_res["query"],
            "matches": [],
            "total_results": 0,
        }


class RAGQueryNode(Node):
    """Node for question answering over retrieved code."""

    def __init__(self, max_retries: int = 1, wait: int = 0) -> None:
        super().__init__(max_retries, wait)
        self.repository = create_repository()
        self.llm = get_default_llm_client()
        self.cfg = get_config()

    def prep(self, shared: Dict[str, Any]) -> Dict[str, Any]:
        project_name = self.params.get("project_name") or shared.get("project_name")
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
        return {"project_name": project_name, "question": question, "limit": limit}

    def exec(self, prep_res: Dict[str, Any]) -> Dict[str, Any]:
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

        context_lines: List[str] = []
        for doc in results:
            context_lines.append(f"File: {doc.path}")
            if hasattr(doc, "start_line") and hasattr(doc, "end_line"):
                context_lines.append(f"Lines: {doc.start_line}-{doc.end_line}")
            context_lines.append("Content:")
            context_lines.append(doc.content)
            context_lines.append("")
        context = "\n".join(context_lines)

        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert code assistant. Use the provided code context to answer "
                    "the user's question accurately and concisely. If the context doesn't contain "
                    "relevant information, say so."
                ),
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
            "context_sources": [
                {"file": doc.path, "score": getattr(doc, "score", 0.0)} for doc in results
            ],
            "action": "query",
        }

    def exec_fallback(self, prep_res: Dict[str, Any], exc: Exception) -> Dict[str, Any]:
        return {
            "status": "error",
            "message": f"RAG query failed: {exc}",
            "project_name": prep_res["project_name"],
            "question": prep_res["question"],
            "answer": "Sorry, I couldn't process your question due to an error.",
        }


class RAGFlow(Flow):
    """Complete RAG workflow for codebase analysis."""

    def __init__(self) -> None:
        super().__init__()
        self.index_node = RAGIndexNode()
        self.search_node = RAGSearchNode()
        self.query_node = RAGQueryNode()

        self.start(self.index_node)
        self.index_node.next(self.search_node, "search")
        self.index_node.next(self.query_node, "query")

        self.nodes: Dict[str, Node] = {
            "index": self.index_node,
            "search": self.search_node,
            "query": self.query_node,
        }

    def prep(self, shared: Dict[str, Any]) -> Dict[str, Any]:
        action = self.params.get("action") or shared.get("action")
        if not action:
            raise ValueError("action parameter is required (index, search, or query)")

        if action == "index":
            if not self.params.get("project_path") and not shared.get("project_path"):
                raise ValueError("project_path is required for indexing")
        else:
            if not self.params.get("project_name") and not shared.get("project_name"):
                raise ValueError("project_name is required for search/query")
            if action == "search" and not (
                self.params.get("query") or shared.get("query")
            ):
                raise ValueError("query is required for search")
            if action == "query" and not (
                self.params.get("question")
                or self.params.get("query")
                or shared.get("question")
                or shared.get("query")
            ):
                raise ValueError("question/query is required for query")
        return {"action": action}

    def _orch(self, shared: Dict[str, Any], params: Optional[Dict[str, Any]] = None) -> Any:
        action = self.params.get("action") or shared.get("action") or "index"
        node = self.nodes.get(action)
        if not node:
            raise ValueError(f"Unknown action: {action}")
        node.set_params(params or {**self.params})
        return node._run(shared)

    def post(self, shared: Dict[str, Any], prep_res: Dict[str, Any], exec_res: Any) -> Any:
        return exec_res


def create_rag_flow() -> RAGFlow:
    return RAGFlow()


def run_rag_workflow(action: str, **kwargs: Any) -> Dict[str, Any]:
    flow = create_rag_flow()
    params = {"action": action, **kwargs}
    return flow.run(params)


__all__ = [
    "RAGFlow",
    "RAGIndexNode",
    "RAGSearchNode",
    "RAGQueryNode",
    "create_rag_flow",
    "run_rag_workflow",
]
