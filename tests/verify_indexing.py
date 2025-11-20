
import os
import shutil
import time
from pathlib import Path
from retrieval.codebase_indexer import SemanticCodeIndexer
from retrieval.index import create_index

def setup_dummy_project(root: Path):
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    
    # Create some dummy python files
    for i in range(10):
        (root / f"file_{i}.py").write_text(f"def func_{i}():\n    print('Hello from {i}')\n", encoding="utf-8")

def verify_indexing():
    root = Path("temp_test_project").resolve()
    setup_dummy_project(root)
    
    print(f"Testing indexing on {root}...")
    
    indexer = SemanticCodeIndexer(batch_size=2)
    
    start_time = time.time()
    index = indexer.ensure_index(root, refresh=True, show_progress=True)
    end_time = time.time()
    
    print(f"Indexing completed in {end_time - start_time:.2f} seconds")
    print(f"Project Key: {index.project_key}")
    print(f"Collection: {index.collection_name}")
    
    # Verify search
    hits, _ = indexer.search(root, "Hello", limit=5)
    print(f"Search found {len(hits)} hits")
    for hit in hits:
        print(f" - {hit.chunk.relative_path}: {hit.score:.4f}")

    # Cleanup
    indexer.close()
    shutil.rmtree(root)

if __name__ == "__main__":
    verify_indexing()
