# ABOUTME: Template-engine regression fixture: REDUCE expansions over species and reaction properties
# ABOUTME: Rendered output is pinned in tests/snapshots/template_engine/

# $$JAFF REDUCE specie_charge IN specie_charges
total_charge = ((0) + (-1) + (1) + (2) + (1))
scaled = 2 * ((0 * 3) + (-1 * 3) + (1 * 3) + (2 * 3) + (1 * 3)) + 1
not_a_reduction = $specie_charge$
# $$JAFF END

# $$JAFF REDUCE specie_mass, specie_charge IN specie_masses, specie_charges
mass_weighted_charge = ((1.673773e-24 * 0) + (9.10938371e-28 * -1) + (1.6728620616289998e-24 * 1) + (6.644651123258e-24 * 2) + (6.645562061629e-24 * 1))
# $$JAFF END

# $$JAFF REDUCE charge_truth IN charge_truths
n_charged = ((0) + (1) + (1) + (1) + (1))
# $$JAFF END

# $$JAFF REDUCE photo_reaction_truth IN photo_reaction_truths
n_photo = ((0) + (0) + (0))
# $$JAFF END

# $$JAFF REDUCE charged_specie_charge_ne IN charged_specie_charges_ne
ion_charge = ((1) + (2) + (1))
# $$JAFF END

# $$JAFF REDUCE neutral_specie_mass IN neutral_specie_masses { REPLACE { "e-(\d+)" -> "E-\1" } }
neutral_mass = ((1.673773E-24))
# $$JAFF END
