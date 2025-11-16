from __future__ import annotations

"""Helpers for loading official tree-sitter language bindings."""

from importlib import import_module
from typing import Callable, Dict, Tuple

from tree_sitter import Language

# Map tree-sitter language identifiers to python packages and attribute names.
_LANGUAGE_SOURCES: Dict[str, Tuple[str, str]] = {
    "python": ("tree_sitter_python", "language"),
    "javascript": ("tree_sitter_javascript", "language"),
    "typescript": ("tree_sitter_typescript", "language_typescript"),
    "tsx": ("tree_sitter_typescript", "language_tsx"),
    "go": ("tree_sitter_go", "language"),
    "java": ("tree_sitter_java", "language"),
    "rust": ("tree_sitter_rust", "language"),
    "c": ("tree_sitter_c", "language"),
    "cpp": ("tree_sitter_cpp", "language"),
    "c_sharp": ("tree_sitter_c_sharp", "language"),
    "php": ("tree_sitter_php", "language"),
    "ruby": ("tree_sitter_ruby", "language"),
    "swift": ("tree_sitter_swift", "language"),
    "kotlin": ("tree_sitter_kotlin", "language"),
}

_LANGUAGE_CACHE: Dict[str, Language] = {}


def is_language_supported(language: str) -> bool:
    """Return True if the language has a configured python binding."""

    return language in _LANGUAGE_SOURCES


def load_language(language: str) -> Language:
    """Load a tree-sitter Language object for the given identifier.

    Raises:
        KeyError: if the language is not configured.
        ImportError: if the corresponding python package is not installed.
        AttributeError: if the module does not expose the expected attribute.
    """

    if language in _LANGUAGE_CACHE:
        return _LANGUAGE_CACHE[language]

    module_name, attr_name = _LANGUAGE_SOURCES[language]
    module = import_module(module_name)
    factory: Callable[[], object] = getattr(module, attr_name)
    ts_language = factory()
    lang = Language(ts_language)
    _LANGUAGE_CACHE[language] = lang
    return lang


def available_languages() -> Dict[str, str]:
    """Return a mapping of supported language identifiers to module names."""

    return {language: module for language, (module, _) in _LANGUAGE_SOURCES.items()}
