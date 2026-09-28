"""Rho/tau builders accept a loaded catalogue in place of a path."""

import numpy as np
import pytest
from astropy.io import fits

from shear_psf_leakage.rho_tau_stat import Catalogs, RhoStat

PARAMS = {
    "ra_PSF_col": "RA",
    "dec_PSF_col": "DEC",
    "e1_PSF_col": "HSM_G1_PSF",
    "e2_PSF_col": "HSM_G2_PSF",
    "e1_star_col": "HSM_G1_STAR",
    "e2_star_col": "HSM_G2_STAR",
    "PSF_size": "HSM_T_PSF",
    "star_size": "HSM_T_STAR",
    "patch_number": 2,
    "ra_units": "deg",
    "dec_units": "deg",
}


@pytest.fixture
def star_cat():
    rng = np.random.default_rng(1)
    n = 200
    columns = [PARAMS[key] for key in PARAMS if key.endswith(("_col", "size"))]
    cat = np.empty(n, dtype=[(name, "f8") for name in columns])
    cat["RA"] = rng.uniform(10, 11, n)
    cat["DEC"] = rng.uniform(-1, 1, n)
    for name in ("HSM_G1_PSF", "HSM_G2_PSF", "HSM_G1_STAR", "HSM_G2_STAR"):
        cat[name] = rng.normal(0, 0.05, n)
    cat["HSM_T_PSF"] = rng.uniform(0.5, 0.6, n)
    cat["HSM_T_STAR"] = rng.uniform(0.5, 0.6, n)
    return cat


def test_read_shear_cat_returns_loaded_table(star_cat):
    catalogs = Catalogs(params=PARAMS)
    assert catalogs.read_shear_cat(None, star_cat) is star_cat
    assert catalogs.read_shear_cat(star_cat, None) is star_cat


def test_rho_catalogues_from_table_match_path(star_cat, tmp_path):
    path = tmp_path / "stars.fits"
    fits.BinTableHDU(star_cat).writeto(path)

    built = {}
    for label, source in (("path", str(path)), ("table", star_cat)):
        rho = RhoStat(params=dict(PARAMS))
        rho.build_cat_to_compute_rho(source, catalog_id=label)
        built[label] = rho.catalogs

    for kind in ("psf", "psf_error", "psf_size_error"):
        from_path = built["path"].get_cat(f"{kind}_path")
        from_table = built["table"].get_cat(f"{kind}_table")
        np.testing.assert_array_equal(from_path.g1, from_table.g1)
        np.testing.assert_array_equal(from_path.g2, from_table.g2)
