# Parser format-ownership refactor: bucketed two-phase parsing

**Date:** 2026-09-19
**Branch:** bug-fix/semi-exhaustive (worktree bug-fix-exhaustive)
**Status:** design approved, pending spec review
**Sequencing:** land this FIRST; the density-alias grammar migration
(`2026-09-19-density-alias-grammar-design.md`) follows on top.

## Problem

The engine (`NetworkParser`) is coupled to specific formats:

- `__set_known_replacments` hardcodes KROME/PRIZMO shorthand aliases in the
  engine, so format plugins are not self-contained.
- Reaction `handle()` runs inline per line, so a format cannot batch-process its
  own reactions — which blocks features like UCLCHEM special reactions (one
  input line expanding into several reactions).
- There is no explicit, stable reaction identity: `.jfunc` `chemRateN` /
  `deltaEN` / `deltaRadN` and `rc_N` bind to a reaction *number*, but that
  number is only implicit (`enumerate` position), and `rc_N` currently uses the
  post-dedup catalogue position while `chemRateN` uses the file-order position —
  an inconsistency that misbinds when `duplicate_policy` merges reactions.

## Goal

Make each format own its parsing and format-local normalization, via a
two-phase bucketed pipeline, while keeping global standardization in `Network`.
Introduce explicit reaction identities. Change **no** network-format file and
break **no** existing behavior (guarded by characterization tests).

## Architecture: four phases

### Phase 1 — detect & bucket (engine)

For each non-blank line, match formats by ascending `priority` (unchanged).

- **Directive formats** (`krome_format` p10, `krome_var` p20, `prizmo_vars`
  p30) keep `handle()` **inline** — they mutate live state / populate
  `ctx.globals`, which later phases depend on.
- **Reaction formats** (`prizmo` p40, `udfa` p50, `krome` p60, `uclchem` p70,
  `kida` p80) no longer handle inline. The engine:
  1. increments a global `source_index` counter (0-based, one per matched
     reaction line);
  2. calls `fmt.capture(match, ctx)` → a `RawReaction` record carrying a
     **metadata snapshot** of the live state the format depends on (e.g. KROME
     `nreact`/`nprod`), so deferred processing never reads stale state;
  3. appends the record to `bucket[fmt.name]`.

A class attribute `emits_reactions: bool = False` on `NetworkFormat`
(overridden `True` on the five reaction formats) tells the engine which branch
to take.

**Why the snapshot matters:** COthin has 30 `@format` headers and popsicle 15
(+4 `@ice`); column counts change mid-file. Batch-processing without a per-line
snapshot would make every reaction see the last header's counts.

### Phase 2 — per-format processing (format-side)

The engine hands each reaction format its bucket:
`fmt.process(records, ctx) -> list[parsedListProps]`. This runs today's
`handle()` body, but:

- reads `record.metadata` instead of live `ctx.state`;
- does all **format-local** normalization: species token maps, `f90_convert`,
  shorthand handling, and rate-string → SymPy sympification against
  `ctx.globals` (already fully populated by phase 1 directives);
- tags each emitted `parsedListProps` with the record's `source_index`.

A format may emit **more than one** reaction per record (UCLCHEM special
reactions): each emitted reaction shares the record's `source_index`; a
per-record sub-order disambiguates them for the final sort.

### Phase 3 — merge & sort (engine)

Concatenate every format's phase-2 output, sort by `(source_index, sub_order)`
→ the file-ordered parsed list, identical in shape and order to today's
`parsed_list` (plus the new `source_index` field). `get_parsed()` returns it.

### Phase 4 — Network (essentially unchanged)

`Network.__load_network` iterates the parsed list and:

- builds the species catalogue in file order (species indices unchanged);
- binds aux funcs using **`record.source_index`** instead of `enumerate i`
  (behaviour-identical today, since they are equal, but now explicit and
  dedup-proof): `chemrate{source_index}`, `deltae{source_index}`,
  `deltarad{source_index}`;
- constructs `Reaction(index=source_index, …)` — `Reaction.index` keeps its
  current meaning (file-side number), so codegen `reaction_idx` is unaffected;
- runs global standardization (`_standardize_symbols`: `n_X`→`nden`, element
  sums, `chi_pe`) and dedup/merge (`duplicate_policy`, rate-segments).

`catalogue_index` = the reaction's position in the final `Reactions` catalogue
(post-dedup); it is what the reactant/product matrices, flux, and ODE indexing
already use. It is derived from catalogue construction, not stored on the record.

## Explicit indices

| name | meaning | drives |
|---|---|---|
| `source_index` | reaction's file-side number (0-based, per matched line, stable under dedup) | `.jfunc` `chemRateN`/`deltaEN`/`deltaRadN`, `Reaction.index`, `rc_N` |
| `catalogue_index` | position among valid, deduped, sorted reactions | reactant/product matrices, flux, ODE |

## `rc_N` fix (behaviour change, intended)

`rc_N` currently resolves `self.reactions[num]` by **catalogue position**
(`network.py:1199`), inconsistent with `chemRateN` (file-side). Change `rc_N`
to resolve by **`source_index`**, matching `chemRateN`, so both are dedup-safe.
Add a regression test with a network where dedup shifts positions.

## Pitfalls / constraints (must hold)

1. **Aux binding stays file-side.** `chemRateN`/`deltaEN`/`deltaRadN` bind to
   `source_index`, never `catalogue_index`. GOW `.jfunc` is 0-based
   (`chemRate0` = reaction 0).
2. **Species-discovery order = file order.** Phase 3 restores file order before
   Network builds the catalogue, so species indices and codegen `idx_X`
   assignment are unchanged.
3. **Directives stay inline** so `ctx.globals` and format state are ready before
   phase 2 sympification.
4. **Multi-temp-range merges:** a merged catalogue entry aggregates several
   source lines; each source line keeps its own `source_index` for its
   segment's `chemRateN`. Network's per-source-line loop is preserved.
5. **`__set_known_replacments` relocation** is deferred to a follow-up (moving
   aliases into per-format `replacements()` hooks); this refactor does not
   require it and keeps the engine's current alias map to minimise risk. (The
   earlier per-format-replacements design is subsumed here: once formats own
   `process()`, the alias map is the natural next thing to move format-side.)

## New/changed interfaces

- `NetworkFormat.emits_reactions: bool = False` (class attr).
- `NetworkFormat.capture(self, match, ctx) -> RawReaction` (reaction formats):
  cheap; snapshots needed state; no heavy parsing.
- `NetworkFormat.process(self, records, ctx) -> list[parsedListProps]`
  (reaction formats): the relocated `handle()` body.
- `RawReaction` record class (`_formats/_record.py`): `source_index: int`,
  `line: str`, `nline: int`, `format: str`, `metadata: dict`,
  `sub_order: int = 0`. Extensible for future per-format metadata.
- Directive formats keep `handle()`; the abstract `handle()` becomes optional
  (reaction formats may drop it once `process()` exists) — or `handle()` is kept
  as a thin default that raises for reaction formats to catch misuse.

## Testing (characterization-first; refactor must be behaviour-preserving)

1. **Parser-output snapshot (regression net):** for every bundled network,
   assert `get_parsed()` produces the same `(r, p, tmin, tmax, rate, type)`
   tuples in the same order before and after (the new `source_index` is the only
   addition). Write and pin on the current baseline before refactoring.
2. Keep the existing `tests/test_parser_replacements.py` green throughout.
3. **`source_index` correctness:** a network whose `.jfunc` uses `chemRateN`
   resolves each reaction's custom rate to the correct function after refactor.
4. **`rc_N` file-side:** a network with a dedup-shifted catalogue resolves `rc_N`
   to the file-side reaction, not the shifted catalogue slot (RED before fix).
5. **UCLCHEM special reactions:** deferred to its own spec, but the record's
   `sub_order` must already support one line → many reactions.
6. Full suite + representative codegen goldens (GOW, uclchem, kida) unchanged.

## Out of scope

- Moving `__set_known_replacments` into per-format `replacements()` hooks
  (follow-up; enabled by this refactor).
- UCLCHEM special-reaction expansion (separate spec; this lays the groundwork).
- The `n_X`/`n_X_nuc` density-alias migration (its own spec; lands after).
