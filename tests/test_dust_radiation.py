# ABOUTME: End-to-end test that the dust + radiation modules enable and build
# ABOUTME: on a minimal H photoionization network driven by a jaffgen.toml config

from pathlib import Path
from types import SimpleNamespace

import pytest

from jaff.cli import JaffGen
from jaff.physics import Dust, Radiation

REPO = Path(__file__).resolve().parent.parent
H_PHOTO = REPO / "networks" / "h_photoionization" / "h_photo.jet"

# Mirrors the commented [network.radiation] / [network.dust] blocks shipped in
# src/jaff/templates/generator/microphysics/jaffgen.toml.
CONFIG_TOML = """\
[network.radiation]
bands = [6, 11.2, 13.6]
profile_index = 0
mode = "nph"
rsl = "c_hat"
background_field = "draine"
use_proxy_photoreaction = true

[network.dust]
rv = 3.1
u_reduction = "extinction"
f_reduction = "extinction"
pe_threshold_low = 6
pe_threshold_high = 13.6
"""


def _args(config, outdir):
    """CLI arg namespace driving jaffgen in-process (see test_codegen_e2e)."""
    return SimpleNamespace(
        network=str(H_PHOTO),
        config=str(config),
        label=None,
        funcfile=None,
        duplicate_policy=None,
        replace_nH=None,
        errors=None,
        network_config=None,
        outdir=str(outdir),
        indir=None,
        files=None,
        template="microphysics",
        lang="cxx",
    )


@pytest.fixture
def gen(tmp_path):
    """Run the full jaffgen pipeline with dust + radiation enabled."""
    config = tmp_path / "jaffgen.toml"
    config.write_text(CONFIG_TOML)
    return JaffGen(_args(config, tmp_path / "out"))


def test_radiation_enabled(gen):
    """[network.radiation] with bands builds a Radiation model on the network."""
    assert isinstance(gen.net.radiation, Radiation)


def test_dust_enabled(gen):
    """Presence of [network.dust] builds a Dust model on the network."""
    assert isinstance(gen.net.dust, Dust)


def test_dust_has_photoelectric_emission(gen):
    """The enabled dust model carries its photoelectric-emission sub-model."""
    assert gen.net.dust.pe is not None
