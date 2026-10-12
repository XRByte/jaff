# ABOUTME: Template-engine regression fixture: SUB, GET and HAS directives, REPLACE chains, plain text
# ABOUTME: Rendered output is pinned in tests/snapshots/template_engine/

# Plain comment: passes through untouched.
x = 1  # trailing comment with $nspec$ outside any block stays verbatim

$$JAFF SUB nspec, nelem, nreact, nbands, label, filename, e_idx
n_species = $nspec$
n_species_p1 = $nspec+1$
n_species_m1 = $nspec - 1$
n_species_x2 = $nspec*2$
n_elements = $nelem$
n_reactions = $nreact$
n_bands = $nbands$
label = "$label$"
fname = "$filename$"
idx_e = $e_idx$
untouched = "$nspecx$ $other$"
# a comment inside a SUB block: $nspec$
$$JAFF END

$$JAFF SUB nspec {
    REPLACE {
        "n_" -> "N_"
        "(\d+)" -> "<\1>"
    }
}
n_species = $nspec$
$$JAFF END

$$JAFF GET specie_idx, specie_mass, specie_charge, specie_latex IN H
idx_H = $specie_idx$
idx_H_p1 = $specie_idx+1$
mass_H = $specie_mass$
charge_H = $specie_charge$
latex_H = r"$specie_latex$"
$$JAFF END

$$JAFF GET element_idx IN H
eidx_H = $element_idx$
$$JAFF END

$$JAFF HAS specie H
has_H = $specie$
$$JAFF END

$$JAFF HAS specie CO { REPLACE { "0" -> "False" } }
has_CO = $specie$
$$JAFF END

$$JAFF HAS element He
has_He = $element$
$$JAFF END
