---
tags:
    - Api
    - Code-generation
---

# Codegen

`jaff.codegen.codegen.Codegen`

The `Codegen` class generates source code for chemical reaction networks in multiple target languages. It produces rate coefficients, flux expressions, ODE right-hand sides, and analytical Jacobians using SymPy, with optional common subexpression elimination (CSE).

Supported languages: C++ (`cxx`, `cpp`, `c++`), C (`c`), Fortran 90 (`f90`, `fortran`), Python (`py`, `python`), Rust (`rust`, `rs`), Julia (`julia`, `jl`), R (`r`).

## Constructor

`#!python Codegen(network, lang="c++")`

**Parameters**

**network** : *Network*
:   The chemical reaction network.

**lang** : *str, optional*
:   Target language alias. Default `"c++"`. Resolved to a `Language` via its alias table.

**Raises**

*InvalidLanguageError*
:   If `lang` is not a recognized language.

## Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `net` | `Network` | The reaction network being code-generated |
| `lang` | `Language` | The resolved language config carrying every syntax token (see below) |

### Language tokens (`cg.lang`)

All per-language syntax lives on the `Language` object at `cg.lang`. Per-method
bracket/token overrides (`brac_format`, `matrix_format`, `assignment_op`,
`line_end` on the `get_*_str` methods) are applied by the `scoped_tokens`
decorator, which transiently swaps `cg.lang` for a `Language.derive`-d view for
the duration of that call and restores it afterwards.

| Attribute | Type | Description |
|-----------|------|-------------|
| `lang.name` | `str` | Canonical language identifier (e.g. `"cxx"`, `"fortran"`, `"python"`) |
| `lang.lb`, `lang.rb` | `str` | Left and right bracket characters for 1-D array indexing (e.g. `"["` and `"]"`) |
| `lang.mlb`, `lang.mrb` | `str` | Left and right bracket characters for 2-D array indexing |
| `lang.sep` | `str` | Index separator for 2-D arrays (e.g. `"]["` for C-style, `", "` for Fortran) |
| `lang.assignment_op` | `str` | Assignment operator for the target language (`"="` for most, `"<-"` for R) |
| `lang.line_end` | `str` | Statement terminator for the target language (`";"` for C/C++/Rust, `""` for others) |
| `lang.code_gen` | `Callable` | SymPy printer function used to serialise symbolic expressions |
| `lang.idx_offset` | `int` | Default array index offset (`0` for C/C++/Python/Rust, `1` for Fortran/Julia/R) |
| `lang.comment` | `str` | Single-line comment prefix for the target language (`"//"`, `"!"`, or `"#"`) |
| `lang.types` | `dict` | Mapping from generic type names to language-specific spellings (e.g. `{"double": "double "}` for C++) |
| `lang.extras` | `dict` | Additional language-specific tokens such as type qualifiers and class specifiers |
