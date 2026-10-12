# ABOUTME: Template-engine regression fixture: REPEAT over data lists: vertical/horizontal, offsets, matrices
# ABOUTME: Rendered output is pinned in tests/snapshots/template_engine/

# Vertical expansion: one output line per item.
$$JAFF REPEAT idx, specie IN species
species[$idx$] = "$specie$"
species_1based[$idx+1$] = "$specie$"
offset_down[$idx-1$] = "$specie$"
upper[$IDX$] = "$specie$"
line without placeholders stays once
$$JAFF END

# Horizontal expansion: brackets, quotes and separators come from the template.
$$JAFF REPEAT specie IN species
names_list = ["$specie$", ]
names_tuple = ('$specie$', )
names_set = {$specie$; }
names_bare = [$specie$]
$$JAFF END

$$JAFF REPEAT specie_mass IN specie_masses
masses = [$specie_mass$, ]
$$JAFF END

$$JAFF REPEAT idx, specie_charge IN specie_charges
charges[$idx$] = $specie_charge$
$$JAFF END

$$JAFF REPEAT element IN elements
elements = ["$element$", ]
$$JAFF END

$$JAFF REPEAT idx, reaction IN reactions
reactions[$idx$] = "$reaction$"
$$JAFF END

$$JAFF REPEAT reaction IN reactions
reactions = ["$reaction$", ]
$$JAFF END

# Ragged 2-D lists expand horizontally into nested brackets.
$$JAFF REPEAT reactant IN reactants
reactants = [$reactant$, ]
$$JAFF END

$$JAFF REPEAT product IN products
products = ["$product$", ]
$$JAFF END

# Matrices: vertical needs one $idx$ per dimension.
$$JAFF REPEAT idx, element IN element_density_matrix
density[$idx$][$idx$] = $element$
$$JAFF END

$$JAFF REPEAT element IN element_truth_matrix
truth = [$element$, ]
$$JAFF END

$$JAFF REPEAT idx, tmin IN tmins
tmin[$idx$] = $tmin$
$$JAFF END

$$JAFF REPEAT tmax IN tmaxes
tmax = [$tmax$, ]
$$JAFF END

$$JAFF REPEAT neutral_specie IN neutral_species
neutral = ["$neutral_specie$", ]
$$JAFF END

$$JAFF REPEAT charged_specie IN charged_species
charged = ["$charged_specie$", ]
$$JAFF END

$$JAFF REPEAT idx, charged_specie_index_ne IN charged_specie_indices_ne
charged_ne[$idx$] = $charged_specie_index_ne$
$$JAFF END

$$JAFF REPEAT neutral_specie_mass IN neutral_specie_masses
neutral_masses = [$neutral_specie_mass$, ]
$$JAFF END

$$JAFF REPEAT charge_truth IN charge_truths
charge_truth = [$charge_truth$, ]
$$JAFF END

$$JAFF REPEAT idx, photo_reaction_truth IN photo_reaction_truths
photo[$idx$] = $photo_reaction_truth$
$$JAFF END

$$JAFF REPEAT radiation_boundary IN radiation_boundaries
bands = [$radiation_boundary$, ]
$$JAFF END

$$JAFF REPEAT idx, specie_with_normalized_sign IN species_with_normalized_sign {
    SIGN { POS -> p  NEG -> m }
}
idx_$specie_with_normalized_sign$ = $idx$
$$JAFF END

# REPLACE applies to every generated line, in order.
$$JAFF REPEAT idx, specie IN species {
    REPLACE {
        "\+" -> "_plus"
        "-" -> "_minus"
        "species" -> "SPECIES"
    }
}
species[$idx$] = "$specie$"
$$JAFF END
