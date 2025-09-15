import os
from pathlib import Path
from typing import List, Optional

from qdrant_client import QdrantClient
from config.manager import get_config

# Import from chat-codebase (we'll need to handle the path)
import sys
from .splitter import iter_repository_files, chunk_code_file

# Add chat-codebase to Python path via environment variable
chat_codebase_path = os.getenv("CHAT_CODEBASE_PATH")
if chat_codebase_path and chat_codebase_path not in sys.path:
    sys.path.insert(0, chat_codebase_path)

CHAT_CODEBASE_AVAILABLE = True
try:
    from src.data.repository import Repository as ChatRepository
    from src.model.embedding import OpenAILikeEmbeddingModel
    from src.model.reranker import RerankAPIModel
    from src.data.splitter import Document
    
    # Import settings for default configurations
    from src.config.settings import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
    
    # Define abstract classes for type checking
    class EmbeddingModel:
        pass
    
    class RerankModel:
        pass
    
except ImportError as e:
    CHAT_CODEBASE_AVAILABLE = False
    print(
        "Warning: Could not import chat-codebase components. "
        "Set CHAT_CODEBASE_PATH to the repository root if available.\n"
        f"Details: {e}"
    )
    # Define fallback classes for development with functional indexing/search
    class Document:
        def __init__(self, path: str, content: str, chunk_id: str = None, 
                     score: float = 0.0, start_line: int = 0, end_line: int = 0):
            self.path = path
            self.content = content
            self.chunk_id = chunk_id
            self.score = score
            self.start_line = start_line
            self.end_line = end_line

    class EmbeddingModel:
        pass
    
    class RerankModel:
        pass
    
    class OpenAILikeEmbeddingModel(EmbeddingModel):
        pass
    
    class RerankAPIModel(RerankModel):
        pass

    class ChatRepository:
        """Lightweight in-repo repository with local indexing and simple search.

        This fallback avoids external dependencies while offering usable behavior
        for indexing and searching codebases in development/CI.
        """

        def __init__(self, model=None, vector_client=None, rerank_model=None):
            self.model = model
            self.vector_client = vector_client
            self.rerank_model = rerank_model
            # project_name -> List[Document]
            self._index: dict[str, list[Document]] = {}

        def _iter_files(self, root: Path):
            yield from iter_repository_files(root)

        def _chunk_file(self, path: Path, max_lines: int = 200):
            try:
                chunks = []
                for i, c in enumerate(chunk_code_file(path, max_lines=max_lines), start=0):
                    chunks.append(
                        Document(
                            path=c.path,
                            content=c.content,
                            chunk_id=f"{Path(c.path).name}:{i}",
                            score=0.0,
                            start_line=c.start_line,
                            end_line=c.end_line,
                        )
                    )
                return chunks
            except Exception:
                # Fallback to empty list on unexpected errors
                return []

        def index(self, project_dir: str):
            root = Path(project_dir).expanduser().resolve()
            if not root.exists() or not root.is_dir():
                raise FileNotFoundError(f"Project directory not found: {root}")
            project_name = root.name
            docs: list[Document] = []
            for fpath in self._iter_files(root):
                docs.extend(self._chunk_file(fpath))
            self._index[project_name] = docs
            return {"project_name": project_name, "total_chunks": len(docs)}

        def search(self, project_name: str, query: str, limit: int = 5) -> List[Document]:
            docs = self._index.get(project_name) or []
            if not query:
                return []
            q = query.lower()
            results: list[Document] = []
            for d in docs:
                text = d.content.lower()
                count = text.count(q)
                if count == 0 and q not in d.path.lower():
                    continue
                # simple score: occurrences normalized + path bonus
                norm = count / max(1, len(text))
                path_bonus = 0.1 if q in d.path.lower() else 0.0
                score = norm + path_bonus
                nd = Document(
                    path=d.path,
                    content=d.content,
                    chunk_id=d.chunk_id,
                    score=score,
                    start_line=d.start_line,
                    end_line=d.end_line,
                )
                results.append(nd)
            results.sort(key=lambda x: x.score, reverse=True)
            return results[: max(1, limit)]


class RepositoryAdapter:
    """Adapter for chat-codebase Repository to work with code-agent."""
    
    def __init__(self, 
                 embedding_model: Optional[EmbeddingModel] = None,
                 vector_client: Optional[QdrantClient] = None,
                 rerank_model: Optional[RerankModel] = None,
                 vector_store_path: str = "./storage"):
        """
        Initialize the repository adapter.
        
        Args:
            embedding_model: Embedding model instance
            vector_client: Qdrant client instance
            rerank_model: Reranking model instance
            vector_store_path: Path to vector store
        """
        self.vector_store_path = vector_store_path
        
        # Initialize components with defaults if not provided
        # Use API-based models only (no local models)
        self.embedding_model = embedding_model or OpenAILikeEmbeddingModel()
        # Avoid creating a Qdrant client in fallback mode
        if vector_client is not None:
            self.vector_client = vector_client
        else:
            self.vector_client = QdrantClient(path=vector_store_path) if CHAT_CODEBASE_AVAILABLE else None
        self.rerank_model = rerank_model or RerankAPIModel()
        
        # Create the actual repository instance
        self.repository = ChatRepository(
            model=self.embedding_model,
            vector_client=self.vector_client,
            rerank_model=self.rerank_model
        )
    
    def index_project(self, project_path: str) -> None:
        """
        Index a project directory.
        
        Args:
            project_path: Path to the project directory to index
        """
        project_path = Path(project_path).expanduser().resolve()
        
        if not project_path.exists():
            raise FileNotFoundError(f"Project directory not found: {project_path}")
        
        if not project_path.is_dir():
            raise ValueError(f"Path is not a directory: {project_path}")
        
        print(f"Indexing project: {project_path.name}")
        self.repository.index(str(project_path))
        print(f"Successfully indexed project: {project_path.name}")
    
    def search(self, project_name: str, query: str, limit: int = 5) -> List[Document]:
        """
        Perform semantic search on an indexed project.
        
        Args:
            project_name: Name of the project to search
            query: Search query
            limit: Maximum number of results to return
            
        Returns:
            List of relevant documents
        """
        return self.repository.search(project_name, query, limit)
    
    def format_search_results(self, documents: List[Document]) -> str:
        """
        Format search results for display.
        
        Args:
            documents: List of search result documents
            
        Returns:
            Formatted string of search results
        """
        if not documents:
            return "No results found."
        
        formatted_results = []
        for i, doc in enumerate(documents, 1):
            result = f"{i}. {doc.path}"
            if hasattr(doc, 'score') and doc.score:
                result += f" (score: {doc.score:.3f})"
            if hasattr(doc, 'start_line') and doc.start_line:
                result += f" lines {doc.start_line}-{doc.end_line}"
            formatted_results.append(result)
            
            # Add a preview of the content
            content_preview = doc.content[:200].replace('\n', ' ')
            if len(doc.content) > 200:
                content_preview += "..."
            formatted_results.append(f"   {content_preview}")
            formatted_results.append("")
        
        return "\n".join(formatted_results)


def create_repository() -> RepositoryAdapter:
    """
    Create a RepositoryAdapter instance using centralized configuration.
    
    Returns:
        Configured RepositoryAdapter
    """
    config = get_config()
    rag_config = config.rag
    
    # Use API-based models only
    if rag_config.embedding_model == "openai-like":
        embedding_model = OpenAILikeEmbeddingModel()
    else:
        embedding_model = OpenAILikeEmbeddingModel()
    
    # Use API-based reranking only
    if rag_config.rerank_model == "api":
        rerank_model = RerankAPIModel()
    else:
        rerank_model = RerankAPIModel()
    
    # Create a local vector client only when external chat-codebase is available
    # Use a default on-disk path to avoid extra configuration
    if CHAT_CODEBASE_AVAILABLE:
        vector_client = QdrantClient(path="./storage")
    else:
        vector_client = None
    
    return RepositoryAdapter(
        embedding_model=embedding_model,
        vector_client=vector_client,
        rerank_model=rerank_model,
        vector_store_path="./storage"
    )
