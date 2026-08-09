# ABOUTME: Tests for jaffgen/jaffx CLI wiring of network options
# ABOUTME: --network-config resolution, duplicate_policy plumbing, [network] block extraction

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

FIXTURES = Path(__file__).parent / "fixtures"
DUP = str(FIXTURES / "duplicate_temp_range.dat")
RXN = "H+.H2__H.H2+"


def _bare_jaffgen(network_config=None, duplicate_policy=None):
    """A JaffGen instance with a minimal args namespace, no real CLI parse."""
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
        duplicate_policy=duplicate_policy,
        lang=None,
    )
    return jg


class TestNetworkConfigArg:
    """--network-config resolution onto NetworkArgs.config."""

    def test_typer_callback_maps_flag(self):
        from typer.testing import CliRunner

        import jaff.cli.jaffgen._engine as engine

        captured = {}

        def _capture(self, args):
            captured["args"] = args

        with patch.object(engine.JaffGen, "__init__", _capture):
            runner = CliRunner()
            runner.invoke(engine.app, ["--network", "n", "--network-config", "x.toml"])
            assert captured["args"].network_config == "x.toml"
            runner.invoke(engine.app, ["--network", "n"])
            assert captured["args"].network_config is None

    def test_none_leaves_config_unset(self):
        jg = _bare_jaffgen(network_config=None)
        jg.set_network_options()
        assert jg.state.network_args.config is None

    def test_relative_path_resolves_against_cwd(self, tmp_path, monkeypatch):
        (tmp_path / "networks").mkdir()
        rel = Path("networks") / "jaff.toml"
        (tmp_path / rel).write_text('[network.rates]\nT_cutoff = "clip"\n')
        monkeypatch.chdir(tmp_path)

        jg = _bare_jaffgen(network_config=str(rel))
        jg.set_network_options()
        cfg = jg.state.network_args.config
        assert cfg == (tmp_path / rel).resolve() and cfg.is_absolute()

    def test_missing_path_raises(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        jg = _bare_jaffgen(network_config="does_not_exist.toml")
        with pytest.raises(FileNotFoundError):
            jg.set_network_options()

    def test_directory_path_raises(self, tmp_path, monkeypatch):
        (tmp_path / "networks").mkdir()
        monkeypatch.chdir(tmp_path)
        jg = _bare_jaffgen(network_config="networks")
        with pytest.raises(FileNotFoundError):
            jg.set_network_options()


class TestNetworkBlockExtraction:
    """[network.*] parses into the shapes fed to metadata slots."""

    def test_block_shapes(self, tmp_path):
        from jaff.drivers import Toml

        body = (
            '[network]\nlabel = "demo"\n'
            '[network.rates]\nT_cutoff = "extrapolate"\n'
            f'[network.reactions."{RXN}"]\nT_cutoff = "clip"\n'
            f'[network.reactions."{RXN}".shielding]\ntype = "leiden"\n'
        )
        p = tmp_path / "jaffgen.toml"
        p.write_text(body)
        network_cfg = Toml(str(p)).get_key("network") or {}

        assert (network_cfg.get("rates") or {}).get("T_cutoff") == "extrapolate"
        reactions_cfg = network_cfg.get("reactions") or {}
        assert reactions_cfg[RXN]["T_cutoff"] == "clip"
        assert reactions_cfg[RXN]["shielding"]["type"] == "leiden"
        assert network_cfg.get("label") == "demo"  # scalar still reachable


class TestDuplicatePolicyWiring:
    """jaffgen resolves duplicate_policy from CLI flag and jaffgen.toml."""

    def _bare(self, cli_value=None, toml_value=None):
        jg = _bare_jaffgen(duplicate_policy=cli_value)
        if toml_value is not None:
            jg.state.network_args.duplicate_policy = toml_value
        return jg

    def test_cli_flag_sets_network_args(self):
        jg = self._bare(cli_value="preserve-last")
        jg.set_network_options()
        assert jg.state.network_args.duplicate_policy == "preserve-last"

    def test_cli_flag_overrides_config_value(self):
        jg = self._bare(cli_value="error", toml_value="preserve-last")
        jg.set_network_options()
        assert jg.state.network_args.duplicate_policy == "error"

    def test_no_cli_flag_keeps_config_value(self):
        jg = self._bare(cli_value=None, toml_value="preserve-last")
        jg.set_network_options()
        assert jg.state.network_args.duplicate_policy == "preserve-last"

    def test_reads_key_from_config_file(self, tmp_path):
        from jaff.cli.jaffgen._engine import JaffGen
        from jaff.cli.jaffgen._structs import ResolvedPath, State
        from jaff.drivers import Toml

        cfg = tmp_path / "jaffgen.toml"
        cfg.write_text('[network]\nduplicate_policy = "error"\n')

        jg = JaffGen.__new__(JaffGen)
        jg.state = State()
        jg.state.config_dir = ResolvedPath(tmp_path, tmp_path)
        jg.state.config_raw = Toml(cfg)
        jg.set_state_from_config()
        assert jg.state.network_args.duplicate_policy == "error"


class TestJaffxWiring:
    """jaffx forwards the duplicate_policy flag onto NetworkArgs."""

    def _args(self, duplicate_policy):
        return SimpleNamespace(
            network=DUP,
            funcfile=False,
            label=None,
            replace_nh=None,
            duplicate_policy=duplicate_policy,
        )

    def test_flag_applied(self):
        from jaff.cli.jaffx._engine import JaffX

        jx = JaffX.__new__(JaffX)
        net = jx.get_network(self._args("preserve-last"))
        assert net.spec.duplicate_policy == "preserve-last"

    def test_none_uses_default(self):
        from jaff.cli.jaffx._engine import JaffX

        jx = JaffX.__new__(JaffX)
        net = jx.get_network(self._args(None))
        assert net.spec.duplicate_policy == "preserve-first"
