# ABOUTME: Tests for duplicate_policy resolution and duplicate rate-segment handling
# ABOUTME: Covers Network/NetworkSpec precedence, dedup behaviour, and jaffgen/jaffx CLI wiring

import os
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from jaff import Network
from jaff.errors import ParserError

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")
KIDA = os.path.join(FIXTURES, "sample_kida.dat")
# Two "H+ + e- -> H" lines over the same (10, 10000) range, alpha 1e-13 then
# 9.99e-13.  The surviving coefficient reveals which policy was applied.
DUP = os.path.join(FIXTURES, "duplicate_temp_range.dat")
DUP_RXN = "H+.e-__H"


def _load(fname=KIDA, config=None, duplicate_policy=None):
    with patch("builtins.print"):
        return Network(
            fname,
            funcfile=False,
            config=config,
            duplicate_policy=duplicate_policy,
        )


def _write_toml(tmp_path, body):
    path = tmp_path / "jaff.toml"
    path.write_text(body)
    return str(path)


class TestPolicyResolution:
    """duplicate_policy resolution on NetworkSpec."""

    def test_default_is_preserve_first(self):
        assert _load().spec.duplicate_policy == "preserve-first"

    @pytest.mark.parametrize(
        "policy", ["preserve-first", "preserve-last", "error"]
    )
    def test_explicit_arg(self, policy):
        assert _load(duplicate_policy=policy).spec.duplicate_policy == policy

    def test_invalid_arg_raises(self):
        with pytest.raises(ValueError, match="Invalid duplicate policy"):
            _load(duplicate_policy="bogus")

    def test_jafftoml_source(self, tmp_path):
        cfg = _write_toml(tmp_path, '[network]\nduplicate_policy = "preserve-last"\n')
        assert _load(config=cfg).spec.duplicate_policy == "preserve-last"

    def test_arg_beats_jafftoml(self, tmp_path):
        cfg = _write_toml(tmp_path, '[network]\nduplicate_policy = "preserve-last"\n')
        assert _load(config=cfg, duplicate_policy="error").spec.duplicate_policy == "error"

    def test_invalid_jafftoml_value_raises(self, tmp_path):
        cfg = _write_toml(tmp_path, '[network]\nduplicate_policy = "bogus"\n')
        with pytest.raises(ValueError, match="Invalid duplicate policy"):
            _load(config=cfg)


class TestDuplicateResolution:
    """End-to-end dedup of a reaction duplicated over the same temperature range."""

    def test_preserve_first_keeps_first_coefficient(self):
        rea = _load(DUP, duplicate_policy="preserve-first").reactions[DUP_RXN]
        assert rea.rate_segments.count == 1
        assert "1.00000000000000e-13" in str(rea.rate)

    def test_preserve_last_keeps_last_coefficient(self):
        # Regression: the duplicate carries a *different* rate, so removing the
        # stale segment must target the stored (old) segment, not the new one.
        rea = _load(DUP, duplicate_policy="preserve-last").reactions[DUP_RXN]
        assert rea.rate_segments.count == 1
        assert "9.99000000000000e-13" in str(rea.rate)

    def test_error_raises_parsererror(self):
        with pytest.raises(ParserError, match="Duplicate reaction"):
            _load(DUP, duplicate_policy="error")


class TestJaffgenWiring:
    """jaffgen resolves duplicate_policy from jaffgen.toml and the CLI flag."""

    def _bare(self, cli_value=None, toml_value=None):
        from jaff.cli.jaffgen._engine import JaffGen
        from jaff.cli.jaffgen._structs import State

        jg = JaffGen.__new__(JaffGen)
        jg.state = State()
        jg.args = SimpleNamespace(
            label=None,
            funcfile=None,
            lang=None,
            replace_nH=None,
            errors=None,
            network_config=None,
            duplicate_policy=cli_value,
        )
        if toml_value is not None:
            jg.state.network_args.duplicate_policy = toml_value
        return jg

    def test_cli_flag_sets_network_args(self):
        jg = self._bare(cli_value="preserve-last")
        jg.set_network_options()
        assert jg.state.network_args.duplicate_policy == "preserve-last"

    def test_cli_flag_overrides_config_value(self):
        # jaffgen.toml already populated network_args; CLI flag must win.
        jg = self._bare(cli_value="error", toml_value="preserve-last")
        jg.set_network_options()
        assert jg.state.network_args.duplicate_policy == "error"

    def test_no_cli_flag_keeps_config_value(self):
        jg = self._bare(cli_value=None, toml_value="preserve-last")
        jg.set_network_options()
        assert jg.state.network_args.duplicate_policy == "preserve-last"

    def test_set_state_from_config_reads_key(self, tmp_path):
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

    def test_get_network_applies_flag(self):
        from jaff.cli.jaffx._engine import JaffX

        jx = JaffX.__new__(JaffX)
        args = SimpleNamespace(
            network=DUP,
            funcfile=False,
            label=None,
            replace_nh=None,
            duplicate_policy="preserve-last",
        )
        with patch("builtins.print"):
            net = jx.get_network(args)
        assert net.spec.duplicate_policy == "preserve-last"

    def test_get_network_none_uses_default(self):
        from jaff.cli.jaffx._engine import JaffX

        jx = JaffX.__new__(JaffX)
        args = SimpleNamespace(
            network=DUP,
            funcfile=False,
            label=None,
            replace_nh=None,
            duplicate_policy=None,
        )
        with patch("builtins.print"):
            net = jx.get_network(args)
        assert net.spec.duplicate_policy == "preserve-first"
