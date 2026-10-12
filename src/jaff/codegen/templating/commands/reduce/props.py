# ABOUTME: REDUCE's properties: name -> (placeholder variable, getter of the value list)
# ABOUTME: Add a property here; the REDUCE command validates and expands from this table

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Tuple

if TYPE_CHECKING:
    from ...context import RenderContext

# Property name -> (placeholder variable, getter of the list of values)
_Getter = Callable[["RenderContext"], List[Any]]
PROPERTIES: Dict[str, Tuple[str, _Getter]] = {
    "specie_charges": ("specie_charge", lambda c: c.net.species.charges()),
    "specie_masses": ("specie_mass", lambda c: c.net.species.masses()),
    "charge_truths": ("charge_truth", lambda c: c.net.species.charge_truths()),
    "photo_reaction_truths": (
        "photo_reaction_truth",
        lambda c: c.net.reactions.photo_reaction_truths(),
    ),
    "photo_reaction_indices": (
        "photo_reaction_index",
        lambda c: c.net.reactions.photo_reaction_indices(),
    ),
    "tmins": ("tmin", lambda c: c.net.reactions.tmins()),
    "tmaxes": ("tmax", lambda c: c.net.reactions.tmaxes()),
    "specie_masses_ne": ("specie_mass_ne", lambda c: c.net.species.masses(ne=True)),
    "specie_charges_ne": (
        "specie_charge_ne",
        lambda c: c.net.species.charges(ne=True),
    ),
    "charged_specie_charges_ne": (
        "charged_specie_charge_ne",
        lambda c: c.net.species.charged("charge", ne=True),
    ),
    "charged_specie_charges": (
        "charged_specie_charge",
        lambda c: c.net.species.charged("charge"),
    ),
    "charge_truths_ne": (
        "charge_truth_ne",
        lambda c: c.net.species.charge_truths(ne=True),
    ),
    "neutral_specie_indices": (
        "neutral_specie_index",
        lambda c: c.net.species.neutral("index"),
    ),
    "charged_specie_indices": (
        "charged_specie_index",
        lambda c: c.net.species.charged("index"),
    ),
    "charged_specie_indices_ne": (
        "charged_specie_index_ne",
        lambda c: c.net.species.charged("index", ne=True),
    ),
    "charged_specie_masses_ne": (
        "charged_specie_mass_ne",
        lambda c: c.net.species.charged("mass", ne=True),
    ),
    "neutral_specie_masses": (
        "neutral_specie_mass",
        lambda c: c.net.species.neutral("mass"),
    ),
    "charged_specie_masses": (
        "charged_specie_mass",
        lambda c: c.net.species.charged("mass"),
    ),
}
PROPS: Tuple[str, ...] = tuple(PROPERTIES)
