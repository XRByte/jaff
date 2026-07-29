# ABOUTME: Tests for temperature-cutoff (clip/extrapolate) resolution
# ABOUTME: Covers jaff.toml + jaffgen metadata sources, Model-A precedence, jaffgen wiring

import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from sympy import Max, symbols

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from jaff import Network
from jaff.errors import ParserError

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")
KIDA = os.path.join(FIXTURES, "sample_kida.dat")

# Unique, tgas-dependent reaction in sample_kida.dat.  Under the default "clip"
# behaviour its rate carries a Max(tmin, tgas) clamp; "extrapolate" removes it.
RXN = "H+.H2__H.H2+"


def _load(config=None, metadata=None):
    """Load sample_kida with an optional jaff.toml path and Network metadata."""
    with patch("builtins.print"):
        return Network(
            KIDA,
            funcfile=False,
            _from_cli=True,
            config=config,
            _metadata=metadata or {},
        )


def _is_clipped(net, srxn=RXN):
    """True when the reaction's rate still carries the tgas clamp (clip)."""
    return net.reactions.from_serialized(srxn).rate.has(Max)


def _write_toml(tmp_path, body):
    path = tmp_path / "jaff.toml"
    path.write_text(body)
    return str(path)


class TestTemperatureCutoffResolution:
    """T_cutoff resolution from jaff.toml and jaffgen metadata."""

    def test_default_is_clip(self):
        # No config, no metadata -> built-in default clamps tgas.
        assert _is_clipped(_load()) is True
        assert symbols("tgas") in _load().reactions.from_serialized(RXN).rate.free_symbols

    def test_jafftoml_global_extrapolate(self, tmp_path):
        cfg = _write_toml(tmp_path, '[network.rates]\nT_cutoff = "extrapolate"\n')
        assert _is_clipped(_load(config=cfg)) is False

    def test_jafftoml_per_reaction_overrides_own_global(self, tmp_path):
        # Global clip, per-reaction extrapolate -> per-reaction wins.
        cfg = _write_toml(
            tmp_path,
            '[network.rates]\nT_cutoff = "clip"\n'
            f'[network.reactions."{RXN}"]\nT_cutoff = "extrapolate"\n',
        )
        assert _is_clipped(_load(config=cfg)) is False

    def test_case_insensitive(self, tmp_path):
        cfg = _write_toml(tmp_path, '[network.rates]\nT_cutoff = "EXTRAPOLATE"\n')
        assert _is_clipped(_load(config=cfg)) is False

    def test_invalid_value_raises(self, tmp_path):
        cfg = _write_toml(tmp_path, '[network.rates]\nT_cutoff = "bogus"\n')
        with pytest.raises(ParserError):
            _load(config=cfg)


class TestPrecedenceModelA:
    """jaffgen-local > jaff-local > jaffgen-global > jaff-global > default."""

    def test_jaffgen_local_beats_jaff_local(self, tmp_path):
        # jaff.toml per-reaction clip, jaffgen per-reaction extrapolate -> extrapolate.
        cfg = _write_toml(
            tmp_path, f'[network.reactions."{RXN}"]\nT_cutoff = "clip"\n'
        )
        meta = {"reaction_props": {RXN: {"T_cutoff": "extrapolate"}}}
        assert _is_clipped(_load(config=cfg, metadata=meta)) is False

    def test_jaff_local_beats_jaffgen_global(self, tmp_path):
        # Model A: a per-reaction setting is never overridden by any global.
        cfg = _write_toml(
            tmp_path, f'[network.reactions."{RXN}"]\nT_cutoff = "extrapolate"\n'
        )
        meta = {"rate_props": {"T_cutoff": "clip"}}
        assert _is_clipped(_load(config=cfg, metadata=meta)) is False

    def test_jaffgen_global_beats_jaff_global(self, tmp_path):
        cfg = _write_toml(tmp_path, '[network.rates]\nT_cutoff = "clip"\n')
        meta = {"rate_props": {"T_cutoff": "extrapolate"}}
        assert _is_clipped(_load(config=cfg, metadata=meta)) is False

    def test_reaction_props_tcutoff_without_jaffgen_object(self):
        # T_cutoff-only reaction_props (no shielding) must not require jaffgen_object.
        meta = {"reaction_props": {RXN: {"T_cutoff": "extrapolate"}}}
        assert _is_clipped(_load(metadata=meta)) is False


class TestJaffgenWiring:
    """jaffgen --network-config arg + [network] metadata assembly, in isolation."""

    def _bare_jaffgen(self, network_config=None):
        # Build a JaffGen without running the pipeline, then drive only
        # set_network_options(), which resolves --network-config onto
        # state.network_props.config.
        from types import SimpleNamespace

        from jaff.cli.jaffgen._engine import JaffGen
        from jaff.cli.jaffgen._structs import State

        jg = JaffGen.__new__(JaffGen)
        jg.state = State()
        jg.args = SimpleNamespace(
            label=None,
            funcfile=None,
            replace_nH=None,
            errors=None,
            network_config=network_config,
            lang=None,
        )
        return jg, JaffGen

    def test_network_config_arg_maps(self):
        # The typer callback maps --network-config onto args.network_config,
        # defaulting to None when the flag is absent.
        from typer.testing import CliRunner

        import jaff.cli.jaffgen._engine as engine

        captured = {}

        def _capture(self, args):
            captured["args"] = args

        app = engine.app
        with patch.object(engine.JaffGen, "__init__", _capture):
            runner = CliRunner()
            runner.invoke(app, ["--network", "n", "--network-config", "x.toml"])
            assert captured["args"].network_config == "x.toml"
            runner.invoke(app, ["--network", "n"])
            assert captured["args"].network_config is None

    def test_set_network_config_none(self):
        jg, _ = self._bare_jaffgen(network_config=None)
        jg.set_network_options()
        assert jg.state.network_props.config is None

    def test_set_network_config_resolves_cwd(self, tmp_path, monkeypatch):
        # A relative --network-config resolves against the CWD, so run from a
        # scratch directory rather than relying on the repository layout.
        (tmp_path / "networks").mkdir()
        rel = Path("networks") / "jaff.toml"
        (tmp_path / rel).write_text('[network.rates]\nT_cutoff = "clip"\n')
        monkeypatch.chdir(tmp_path)

        jg, _ = self._bare_jaffgen(network_config=str(rel))
        jg.set_network_options()
        cfg = jg.state.network_props.config
        assert cfg == (tmp_path / rel).resolve() and cfg.is_absolute()

    def test_set_network_config_missing_raises(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        jg, _ = self._bare_jaffgen(network_config="does_not_exist.toml")
        with pytest.raises(FileNotFoundError):
            jg.set_network_options()

    def test_set_network_config_dir_raises(self, tmp_path, monkeypatch):
        (tmp_path / "networks").mkdir()
        monkeypatch.chdir(tmp_path)
        jg, _ = self._bare_jaffgen(network_config="networks")
        with pytest.raises(FileNotFoundError):
            jg.set_network_options()

    def test_network_block_extraction(self, tmp_path):
        # [network.*] parses into the shapes T3 feeds to metadata slots.
        from jaff.drivers import Toml

        body = (
            '[network]\nlabel = "demo"\n'
            "[network.rates]\nT_cutoff = \"extrapolate\"\n"
            f'[network.reactions."{RXN}"]\nT_cutoff = "clip"\n'
            f'[network.reactions."{RXN}".shielding]\ntype = "leiden"\n'
        )
        p = tmp_path / "jaffgen.toml"
        p.write_text(body)
        network_cfg = Toml(str(p)).get_key("network") or {}
        reactions_cfg = network_cfg.get("reactions") or {}
        rates_cfg = network_cfg.get("rates") or {}

        assert rates_cfg.get("T_cutoff") == "extrapolate"
        assert reactions_cfg[RXN]["T_cutoff"] == "clip"
        assert reactions_cfg[RXN]["shielding"]["type"] == "leiden"
        assert network_cfg.get("label") == "demo"  # scalar still reachable
