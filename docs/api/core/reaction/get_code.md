---
tags:
    - Api
    - Reaction
---

# get_code

`#!python get_code(lang="cpp")`

Returns the rate expression as a code string in the target language.

**Parameters**

**lang** : _str, optional_
: Target language name or alias — `"python"`/`"py"`, `"c"`, `"cxx"`/`"cpp"`/`"c++"`, `"fortran"`/`"f90"`, `"rust"`/`"rs"`, `"julia"`/`"jl"`, `"r"`. Default `"cpp"`.

**Returns**

_str_
: Rate expression code. A photo-reaction rate prints as `photorates(n, lo, hi)`, where `n` is the reaction's 1-based position among the network's photo-reactions.

**Raises**

_InvalidLanguageError_
: If `lang` is not a recognized language.
