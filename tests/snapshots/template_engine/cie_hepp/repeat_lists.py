# ABOUTME: Template-engine regression fixture: REPEAT over data lists: vertical/horizontal, offsets, matrices
# ABOUTME: Rendered output is pinned in tests/snapshots/template_engine/

# Vertical expansion: one output line per item.
# $$JAFF REPEAT idx, specie IN species
species[0] = "H"
species[1] = "e-"
species[2] = "H+"
species[3] = "He++"
species[4] = "He+"
species_1based[1] = "H"
species_1based[2] = "e-"
species_1based[3] = "H+"
species_1based[4] = "He++"
species_1based[5] = "He+"
offset_down[-1] = "H"
offset_down[0] = "e-"
offset_down[1] = "H+"
offset_down[2] = "He++"
offset_down[3] = "He+"
upper[0] = "H"
upper[1] = "e-"
upper[2] = "H+"
upper[3] = "He++"
upper[4] = "He+"
line without placeholders stays once
# $$JAFF END

# Horizontal expansion: brackets, quotes and separators come from the template.
# $$JAFF REPEAT specie IN species
names_list = ["H", "e-", "H+", "He++", "He+"]
names_tuple = ('H', 'e-', 'H+', 'He++', 'He+')
names_set = {H; e-; H+; He++; He+}
names_bare = [H, e-, H+, He++, He+]
# $$JAFF END

# $$JAFF REPEAT specie_mass IN specie_masses
masses = [1.673773e-24, 9.10938371e-28, 1.6728620616289998e-24, 6.644651123258e-24, 6.645562061629e-24]
# $$JAFF END

# $$JAFF REPEAT idx, specie_charge IN specie_charges
charges[0] = 0
charges[1] = -1
charges[2] = 1
charges[3] = 2
charges[4] = 1
# $$JAFF END

# $$JAFF REPEAT element IN elements
elements = ["H", "He"]
# $$JAFF END

# $$JAFF REPEAT idx, reaction IN reactions
reactions[0] = "H + e- -> H+ + e- + e-"
reactions[1] = "H+ + e- -> H"
reactions[2] = "He++ + e- -> He+"
# $$JAFF END

# $$JAFF REPEAT reaction IN reactions
reactions = ["H + e- -> H+ + e- + e-", "H+ + e- -> H", "He++ + e- -> He+"]
# $$JAFF END

# Ragged 2-D lists expand horizontally into nested brackets.
# $$JAFF REPEAT reactant IN reactants
reactants = [[H, e-], [H+, e-], [He++, e-]]
# $$JAFF END

# $$JAFF REPEAT product IN products
products = [["H+", "e-", "e-"], ["H"], ["He+"]]
# $$JAFF END

# Matrices: vertical needs one $idx$ per dimension.
# $$JAFF REPEAT idx, element IN element_density_matrix
density[0][0] = 1
density[0][1] = 0
density[0][2] = 1
density[0][3] = 0
density[0][4] = 0
density[1][0] = 0
density[1][1] = 0
density[1][2] = 0
density[1][3] = 1
density[1][4] = 1
# $$JAFF END

# $$JAFF REPEAT element IN element_truth_matrix
truth = [[1, 0, 1, 0, 0], [0, 0, 0, 1, 1]]
# $$JAFF END

# $$JAFF REPEAT idx, tmin IN tmins
tmin[0] = None
tmin[1] = None
tmin[2] = None
# $$JAFF END

# $$JAFF REPEAT tmax IN tmaxes
tmax = [None, None, None]
# $$JAFF END

# $$JAFF REPEAT neutral_specie IN neutral_species
neutral = ["H"]
# $$JAFF END

# $$JAFF REPEAT charged_specie IN charged_species
charged = ["e-", "H+", "He++", "He+"]
# $$JAFF END

# $$JAFF REPEAT idx, charged_specie_index_ne IN charged_specie_indices_ne
charged_ne[0] = 2
charged_ne[1] = 3
charged_ne[2] = 4
# $$JAFF END

# $$JAFF REPEAT neutral_specie_mass IN neutral_specie_masses
neutral_masses = [1.673773e-24]
# $$JAFF END

# $$JAFF REPEAT charge_truth IN charge_truths
charge_truth = [0, 1, 1, 1, 1]
# $$JAFF END

# $$JAFF REPEAT idx, photo_reaction_truth IN photo_reaction_truths
photo[0] = 0
photo[1] = 0
photo[2] = 0
# $$JAFF END

# $$JAFF REPEAT radiation_boundary IN radiation_boundaries
bands = []
# $$JAFF END

# $$JAFF REPEAT idx, specie_with_normalized_sign IN species_with_normalized_sign {
    # SIGN { POS -> p  NEG -> m }
# }
idx_h = 0
idx_em = 1
idx_hp = 2
idx_hepp = 3
idx_hep = 4
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
SPECIES[1] = "e_minus"
SPECIES[2] = "H_plus"
SPECIES[3] = "He_plus_plus"
SPECIES[4] = "He_plus"
# $$JAFF END
