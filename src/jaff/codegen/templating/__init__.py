# ABOUTME: The $$JAFF template engine: grammar, single-pass block engine and commands
# ABOUTME: TemplateParser is the entry point; jaff.codegen re-exports it

from .parser import TemplateParser

__all__ = ["TemplateParser"]
