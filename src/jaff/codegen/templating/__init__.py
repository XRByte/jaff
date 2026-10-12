# ABOUTME: New $$JAFF template engine: grammar, single-pass block engine and commands
# ABOUTME: TemplateParser is the entry point; the old engine in _template_engine.py stays active

from .parser import TemplateParser

__all__ = ["TemplateParser"]
