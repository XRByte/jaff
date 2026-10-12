# ABOUTME: Template-engine regression fixture: REDUCE expansions over species and reaction properties
# ABOUTME: Rendered output is pinned in tests/snapshots/template_engine/

$$JAFF REDUCE specie_charge IN specie_charges
total_charge = $($specie_charge$)$
scaled = 2 * $($specie_charge$ * 3)$ + 1
not_a_reduction = $specie_charge$
$$JAFF END

$$JAFF REDUCE specie_mass, specie_charge IN specie_masses, specie_charges
mass_weighted_charge = $($specie_mass$ * $specie_charge$)$
$$JAFF END

$$JAFF REDUCE charge_truth IN charge_truths
n_charged = $($charge_truth$)$
$$JAFF END

$$JAFF REDUCE photo_reaction_truth IN photo_reaction_truths
n_photo = $($photo_reaction_truth$)$
$$JAFF END

$$JAFF REDUCE charged_specie_charge_ne IN charged_specie_charges_ne
ion_charge = $($charged_specie_charge_ne$)$
$$JAFF END

$$JAFF REDUCE neutral_specie_mass IN neutral_specie_masses { REPLACE { "e-(\d+)" -> "E-\1" } }
neutral_mass = $($neutral_specie_mass$)$
$$JAFF END
