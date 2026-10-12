# ABOUTME: Template-engine regression fixture: SUB, GET and HAS directives, REPLACE chains, plain text
# ABOUTME: Rendered output is pinned in tests/snapshots/template_engine/

# Plain comment: passes through untouched.
x = 1  # trailing comment with $nspec$ outside any block stays verbatim

# $$JAFF SUB nspec, nelem, nreact, nbands, label, filename, e_idx
n_species = 3
n_species_p1 = 4
n_species_m1 = 2
n_species_x2 = 6
n_elements = 3
n_reactions = 1
n_bands = 0
label = "test_jac_dedt"
fname = "directives.py"
idx_e = None
untouched = "$nspecx$ $other$"
# a comment inside a SUB block: 3
# $$JAFF END

# $$JAFF SUB nspec {
    # REPLACE {
        # "n_" -> "N_"
        # "(\d+)" -> "<\1>"
    # }
# }
N_species = <3>
# $$JAFF END

# $$JAFF GET specie_idx, specie_mass, specie_charge, specie_latex IN H
idx_H = 0
idx_H_p1 = 1
mass_H = 1.673773e-24
charge_H = 0
latex_H = r"{\rm H}"
# $$JAFF END

# $$JAFF GET element_idx IN H
eidx_H = 1
# $$JAFF END

# $$JAFF HAS specie H
has_H = 1
# $$JAFF END

# $$JAFF HAS specie CO { REPLACE { "0" -> "False" } }
has_CO = False
# $$JAFF END

# $$JAFF HAS element He
has_He = 1
# $$JAFF END
