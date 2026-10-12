# ABOUTME: Template-engine regression fixture: REPEAT over data lists: vertical/horizontal, offsets, matrices
# ABOUTME: Rendered output is pinned in tests/snapshots/template_engine/

# Vertical expansion: one output line per item.
# $$JAFF REPEAT idx, specie IN species
species[0] = "H"
species[1] = "D"
species[2] = "He"
species_1based[1] = "H"
species_1based[2] = "D"
species_1based[3] = "He"
offset_down[-1] = "H"
offset_down[0] = "D"
offset_down[1] = "He"
upper[0] = "H"
upper[1] = "D"
upper[2] = "He"
line without placeholders stays once
# $$JAFF END

# Horizontal expansion: brackets, quotes and separators come from the template.
# $$JAFF REPEAT specie IN species
names_list = ["H", "D", "He"]
names_tuple = ('H', 'D', 'He')
names_set = {H; D; He}
names_bare = [H, D, He]
# $$JAFF END

# $$JAFF REPEAT specie_mass IN specie_masses
masses = [1.673773e-24, 3.343586e-24, 6.646473e-24]
# $$JAFF END

# $$JAFF REPEAT idx, specie_charge IN specie_charges
charges[0] = 0
charges[1] = 0
charges[2] = 0
# $$JAFF END

# $$JAFF REPEAT element IN elements
elements = ["D", "H", "He"]
# $$JAFF END

# $$JAFF REPEAT idx, reaction IN reactions
reactions[0] = "H + D -> He"
# $$JAFF END

# $$JAFF REPEAT reaction IN reactions
reactions = ["H + D -> He"]
# $$JAFF END

# Ragged 2-D lists expand horizontally into nested brackets.
# $$JAFF REPEAT reactant IN reactants
reactants = [[H, D]]
# $$JAFF END

# $$JAFF REPEAT product IN products
products = [["He"]]
# $$JAFF END

# Matrices: vertical needs one $idx$ per dimension.
# $$JAFF REPEAT idx, element IN element_density_matrix
density[0][0] = 0
density[0][1] = 1
density[0][2] = 0
density[1][0] = 1
density[1][1] = 0
density[1][2] = 0
density[2][0] = 0
density[2][1] = 0
density[2][2] = 1
# $$JAFF END

# $$JAFF REPEAT element IN element_truth_matrix
truth = [[0, 1, 0], [1, 0, 0], [0, 0, 1]]
# $$JAFF END

# $$JAFF REPEAT idx, tmin IN tmins
tmin[0] = None
# $$JAFF END

# $$JAFF REPEAT tmax IN tmaxes
tmax = [None]
# $$JAFF END

# $$JAFF REPEAT neutral_specie IN neutral_species
neutral = ["H", "D", "He"]
# $$JAFF END

# $$JAFF REPEAT charged_specie IN charged_species
charged = []
# $$JAFF END

# $$JAFF REPEAT idx, charged_specie_index_ne IN charged_specie_indices_ne
# $$JAFF END

# $$JAFF REPEAT neutral_specie_mass IN neutral_specie_masses
neutral_masses = [1.673773e-24, 3.343586e-24, 6.646473e-24]
# $$JAFF END

# $$JAFF REPEAT charge_truth IN charge_truths
charge_truth = [0, 0, 0]
# $$JAFF END

# $$JAFF REPEAT idx, photo_reaction_truth IN photo_reaction_truths
photo[0] = 0
# $$JAFF END

# $$JAFF REPEAT radiation_boundary IN radiation_boundaries
bands = []
# $$JAFF END

# $$JAFF REPEAT idx, specie_with_normalized_sign IN species_with_normalized_sign {
    # SIGN { POS -> p  NEG -> m }
# }
idx_h = 0
idx_d = 1
idx_he = 2
# $$JAFF END

# REPLACE applies to every generated line, in order.
# $$JAFF REPEAT idx, specie IN species {
    # REPLACE {
        # "\+" -> "_plus"
        # "-" -> "_minus"
        # "species" -> "SPECIES"
    # }
# }
SPECIES[0] = "H"
SPECIES[1] = "D"
SPECIES[2] = "He"
# $$JAFF END
