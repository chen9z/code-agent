import pytest
from adapters.workspace.semantic_splitter import SemanticSplitter, SpanKind

def test_python_context_extraction():
    code = """
class MyClass:
    def my_method(self):
        x = 1
        return x

def global_func():
    pass
"""
    splitter = SemanticSplitter(language="python", chunk_size=100)
    chunks = splitter.split("test.py", code.encode("utf-8"))
    
    # Expected:
    # 1. Class definition (maybe)
    # 2. Method definition with context "MyClass"
    # 3. Global func with no context
    
    # Note: The splitter might chunk differently depending on size.
    # With small chunk_size, we expect granular chunks.
    
    found_method = False
    found_class = False
    
    for chunk in chunks:
        context = chunk.metadata.get("context", [])
        content = chunk.content.strip()
        
        if "def my_method" in content:
            assert "MyClass" in context
            found_method = True
        
        if "class MyClass" in content and "def my_method" not in content:
            # The class header itself might not have context, or be its own context?
            # Based on implementation: node "class_definition" adds name to context for *children*.
            # The class definition node itself starts *before* it adds to context for children?
            # Actually _chunk_node adds context if node.type is in _node_types.
            # And then passes new_context to children.
            # The span created for the node itself uses new_context?
            # Yes: span = Span(..., context=new_context)
            # So the class definition line should have "MyClass" in context?
            # Let's check implementation:
            # if node.type in _node_types: new_context = context + (name,)
            # span = Span(..., context=new_context)
            # So yes, the class def line has context "MyClass".
            assert "MyClass" in context
            found_class = True
            
    assert found_method
    assert found_class

def test_nested_context():
    code = """
class Outer:
    class Inner:
        def method(self):
            pass
"""
    splitter = SemanticSplitter(language="python", chunk_size=1000)
    chunks = splitter.split("test.py", code.encode("utf-8"))
    
    for chunk in chunks:
        if "def method" in chunk.content:
            context = chunk.metadata.get("context", [])
            assert "Outer" in context
            assert "Inner" in context
            # Context includes the method itself
            assert context == ["Outer", "Inner", "method"]
