# ABOUTME: j/k charge-symbol convention — encoding, reverse map, decode, collisions.
import pytest


def test_normalized_names_defaults_are_jk(make_network):
    net = make_network(
        [
            "He + He+ -> He+ + He+ [10,1000] 1e-10",   # forces He, He+ into species
            "He+ + He+ -> He++ + He [10,1000] 1e-10",  # forces He++
        ]
    )
    names = set(net.species.normalized_names())
    assert "hejj" in names          # He++  -> multi-charge replaces ALL
    assert "hej" in names           # He+
    assert "he" in names            # He (neutral, unchanged)


def test_normalized_names_no_collision_metal_vs_anion(make_network):
    # Sn (tin, neutral) vs S- (sulfur anion) must stay distinct under j/k.
    net = make_network(["Sn + S- -> Sn + S- [10,1000] 1e-10"])
    names = list(net.species.normalized_names())
    assert names.count("sn") == 1   # only tin
    assert "sk" in names            # S- -> sk, NOT sn


def test_charge_reverse_map_round_trips(make_network):
    net = make_network(["He + He+ -> He+ + He+ [10,1000] 1e-10"])
    rmap = net.species.charge_reverse_map()
    assert rmap["hej"].name == "He+"
    assert rmap["he"].name == "He"


def test_charge_reverse_map_raises_on_case_collision(make_network):
    # CO and Co both lower-case to "co" -> the map cannot be built.
    net = make_network(["CO + Co -> CO + Co [10,1000] 1e-10"])
    with pytest.raises(ValueError):
        net.species.charge_reverse_map()


def test_decode_multi_charge_density(make_network):
    import sympy
    net = make_network(
        [
            "He+ + He+ -> He++ + He [10,1000] 1e-10",
            "He++ + E -> He+ [10,1000] 1e-10",
        ]
    )
    expr = net._standardize_symbols(sympy.Symbol("n_Hejj"), True)
    idx = net.species["He++"].index
    assert expr == net.ndens[sympy.Idx(idx)]


def test_decode_single_cation(make_network):
    import sympy
    net = make_network(["C + C+ -> C+ + C [10,1000] 1e-10"])
    expr = net._standardize_symbols(sympy.Symbol("n_Cj"), True)
    idx = net.species["C+"].index
    assert expr == net.ndens[sympy.Idx(idx)]


def test_decode_neutral_zero_suffix(make_network):
    import sympy
    net = make_network(["O + O -> O + O [10,1000] 1e-10"])
    expr = net._standardize_symbols(sympy.Symbol("n_O0"), True)
    idx = net.species["O"].index
    assert expr == net.ndens[sympy.Idx(idx)]


def test_decode_neutral_h_vs_sum(make_network):
    import sympy
    net = make_network(["H + H+ -> H+ + H [10,1000] 1e-10"])
    expr = net._standardize_symbols(sympy.Symbol("n_H0"), True)
    idx = net.species["H"].index
    assert expr == net.ndens[sympy.Idx(idx)]


def test_decode_electron(make_network):
    import sympy
    net = make_network(["H -> H+ + e- [10,1000] 1e-10"])
    expr = net._standardize_symbols(sympy.Symbol("n_e"), True)
    idx = net.species["e-"].index
    assert expr == net.ndens[sympy.Idx(idx)]
