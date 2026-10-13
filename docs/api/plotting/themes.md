---
tags:
    - Api
    - Plotting
---

# Themes

A theme has two independent halves, each chosen by name or plain dict.

## Style

Everything except colour: rcParams, panel size, panel gap, axes look, tick
geometry, exponent and inside tick labels, the line-style / marker `cycle`,
line width, marker size and spacing, legend and tag formatting, the legend
size cap, save dpi.

| Name      | Look                                                                                         |
| --------- | -------------------------------------------------------------------------------------------- |
| `"paper"` | Default. Print size (1.9 × 1.6 in panels, 0.08 in gaps); inside y labels; exponent log axes. |
| `"talk"`  | Large fonts and ticks, thick lines (5 × 4 in panels, 0.25 in gaps); `10^k` log labels.       |

A dict is a partial override on a base style (default `"paper"`). Nested
fields (`rc`, `ticks`, `marker`, `legend`, `tag`, `save`) merge key-wise;
others replace:

```python
plot_rates(r, style={"base": "talk", "lw": 1.2, "cycle": ["-", "o"],
                     "rc": {"font.size": 14}})
```

Fields: `rc`, `panel_size` (inches per panel box), `gap` (inches between
panels, rows and columns alike; `0` makes panels touch), `surface`, `ink`,
`ticks`, `exp_ticks`, `inside_ticks` (y tick labels inside the panel),
`cycle`, `lw`, `marker` (`size`, `gap_pt`, `line_gap_pt` for line+marker
styles), `legend`, `legend_max`, `solo_color`, `tag` (`fontsize`, `alpha`;
also used for inside titles), `save` (`dpi`, `pad_inches`).

`ref_cols` (positive int; `"paper"` 4, `"talk"` 3) is the column count a
style is sized for. A figure with more used columns (after empty ones
collapse) has every visual size — fonts, line widths, ticks, markers, gaps,
pads — scaled by `columns / ref_cols`, except panel titles and tags, while
`panel_size` stays: a 6-column
paper figure shrunk to the page reads like a 4-column one. An explicit
`figsize` turns the scaling off. Registered styles are never modified.

`solo_color` (default `#00798c`, teal, in both built-ins) colours an
unlabelled series that is alone on its panel, e.g. a single rate titled with
its equation; when the shared legend already uses that colour, the next
unused palette colour is taken.

`legend_max` (positive int, default 12 in both built-ins) caps legend size: a
legend with more entries is not drawn, and `plot_rates` / `plot_xsecs` without
a `grid` split more curves than this over an automatic grid.

With `exp_ticks`, log axes show at most ~4 labelled decades with minor
ticks on decades, at most 4 between labels (no 2–9 sub-ticks); a log axis
spanning less than 1.5 decades is labelled at fractional exponents (`1.05`, `1.1`, `−1.9`) instead of a lone
decade.

```python
plot(spec | {"style": {"base": "talk", "gap": 0}})   # fused panels, as in a KDE + ECDF pair
```

## Palette

| Name          | Colours                                     |
| ------------- | ------------------------------------------- |
| `"jaff"`      | Default: Okabe-Ito without yellow + distinct GOW hues (14) |
| `"okabe-ito"` | Colour-blind safe (8)                       |
| `"gow"`       | 15 distinct species hues                    |
| `"logo"`      | Logo colours (4)                            |
| `"sunset"`    | Navy to peach (4)                           |
| `"vivid"`     | Red, blue, green, purple, grey              |
| `"muted"`, `"deep"` | seaborn-style cycles                  |

A list is a custom cycle; a dict maps **labels** to colours (other labels use
the default cycle):

```python
plot(spec | {"palette": {"Gong+2017": "#4d4d4d", "1 band": "#0072b2"}})
```

## Cycling

Series without an explicit `color` / `style` are assigned **colour first,
then style**: series *i* gets `palette[i % n]` and `cycle[i // n]`, so dashes
and markers appear only once the colours run out (built-in cycle: four
linestyles, then markers `o X P D s ^`). A style entry that is a marker draws
hollow markers spaced evenly along the curve; a fmt string such as `"-o"` or
`"--D"` draws the line plus sparse hollow markers. The index counts only
series without an explicit colour (resp. style, or a label-mapped colour);
bar series don't use the style cycle, and the style cycle stays on its last
entry once exhausted.

When a shared figure legend keys several panels, the look is assigned per
**label** across the whole figure, so one label has one colour and style in
every panel; unlabelled series continue the cycle after the legend's looks, so
they never resemble a legend entry. A series with `"role": "reference"` is outside both cycles: dark
grey `#4d4d4d`, 1.9 × the style line width, drawn beneath the other series.

## Registration

```python
from jaff.plotting import register_palette, register_style

register_style("poster", {"base": "talk", "rc": {"font.size": 26}})
register_palette("sources", {"Gong+2017": "#4d4d4d", "sim": "#d55e00"})
plot_rates(r, style="poster", palette="sources")
```

Built-in names cannot be overwritten. Rendering applies the style inside
`matplotlib.pyplot.rc_context`; global `rcParams` are never changed.
