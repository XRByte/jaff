---
tags:
    - Api
    - Reaction
---

# Reactions

`jaff.core.reaction.Reactions`

The `Reactions` class is a typed, ordered `Catalogue` of `Reaction` objects. It supports dictionary-style lookup by verbatim string or serialized key, and provides vector accessors for bulk retrieval of reaction properties.

String indexing (`reactions[key]`) is *scalar-or-list*: it returns a single `Reaction` when the key is unique, or a `list[Reaction]` when several mechanisms share that serialized form or verbatim string (a missing key raises `KeyError`). Use the `(id, type)` tuple form to always pick exactly one, or [`from_serialized`](from_serialized.md) / [`all`](all.md) for a result that is always a collection.

## Attributes

**count** : _int_
: Total number of reactions in the collection.

## Constructor

`#!python Reactions(reactions=None)`

**Parameters**

**reactions** : _list\[Reaction\] or None, optional_
: Initial reactions. Default `None` (empty collection).
