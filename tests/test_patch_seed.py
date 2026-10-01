"""Reproducibility of the k-means jackknife patches in rho statistics."""

import numpy as np
import treecorr
from astropy.table import Table

from shear_psf_leakage.rho_tau_stat import RhoStat, TauStat

TREECORR_CONFIG = {
    "ra_units": "deg",
    "dec_units": "deg",
    "sep_units": "arcmin",
    "min_sep": 1.0,
    "max_sep": 60.0,
    "nbins": 5,
    "var_method": "jackknife",
}


def _star_catalogue(path, n=3000, seed=0):
    rng = np.random.default_rng(seed)
    e1_psf = rng.normal(0, 0.02, n)
    e2_psf = rng.normal(0, 0.02, n)
    t_psf = rng.uniform(0.9, 1.1, n)
    Table(
        {
            "RA": rng.uniform(150, 154, n),
            "Dec": rng.uniform(0, 4, n),
            "HSM_G1_PSF": e1_psf,
            "HSM_G2_PSF": e2_psf,
            "HSM_G1_STAR": e1_psf + rng.normal(0, 0.005, n),
            "HSM_G2_STAR": e2_psf + rng.normal(0, 0.005, n),
            "HSM_T_PSF": t_psf,
            "HSM_T_STAR": t_psf * rng.normal(1, 0.01, n),
        }
    ).write(path)
    return path


def _rho(path, output, patch_seed=None):
    params = {
        "ra_PSF_col": "RA",
        "dec_PSF_col": "Dec",
        "e1_PSF_col": "HSM_G1_PSF",
        "e2_PSF_col": "HSM_G2_PSF",
        "e1_star_col": "HSM_G1_STAR",
        "e2_star_col": "HSM_G2_STAR",
        "PSF_size": "HSM_T_PSF",
        "star_size": "HSM_T_STAR",
        "patch_number": 10,
        "ra_units": "deg",
        "dec_units": "deg",
    }
    if patch_seed is not None:
        params["patch_seed"] = patch_seed
    rho = RhoStat(params=params, output=str(output), treecorr_config=TREECORR_CONFIG)
    rho.build_cat_to_compute_rho(str(path), catalog_id="t")
    rho.compute_rho_stats("t", "rho_t.fits")
    cat = rho.catalogs.get_cat("psf_t")
    return cat.patch_centers, cat.patch, rho.rho_stats


def test_patches_and_rho_reproducible(tmp_path):
    # Multithreaded k-means sums in a varying order, so centres agree to
    # rounding rather than bitwise; the patch assignment is exact.
    path = _star_catalogue(tmp_path / "stars.fits")

    centers_a, patch_a, rho_a = _rho(path, tmp_path)
    centers_b, patch_b, rho_b = _rho(path, tmp_path)
    np.testing.assert_allclose(centers_a, centers_b, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(patch_a, patch_b)
    for name in rho_a.dtype.names:
        np.testing.assert_allclose(rho_a[name], rho_b[name], rtol=1e-10, atol=0)

    centers_c, _, _ = _rho(path, tmp_path, patch_seed=7)
    assert not np.allclose(centers_a, centers_c, rtol=0, atol=1e-6)


def test_rho_tau_independent_of_default_tree_depth(tmp_path, monkeypatch):
    # Simulate the defaults on 24- and 48-CPU nodes without using those CPUs.
    determine_top = treecorr.Field._determine_top
    default_top = 5

    def node_default(self, min_top, max_top):
        return determine_top(
            self, default_top if min_top is None else min_top, max_top
        )

    monkeypatch.setattr(treecorr.Field, "_determine_top", node_default)
    config = {**TREECORR_CONFIG, "min_top": 6}
    stars = _star_catalogue(tmp_path / "stars.fits", n=5000)
    rng = np.random.default_rng(10)
    galaxies = Table(
        {
            "RA": rng.uniform(150, 154, 6000),
            "Dec": rng.uniform(0, 4, 6000),
            "e1": rng.normal(0, 0.2, 6000),
            "e2": rng.normal(0, 0.2, 6000),
            "w": np.ones(6000),
        }
    )

    def measure():
        rho = RhoStat(output=str(tmp_path), treecorr_config=config)
        rho.catalogs._params["patch_number"] = 150
        rho.build_cat_to_compute_rho(stars, "t")
        rho.compute_rho_stats("t", "rho.fits")
        tau = TauStat(
            catalogs=rho.catalogs, output=str(tmp_path), treecorr_config=config
        )
        tau.build_cat_to_compute_tau(galaxies, "gal", "t")
        tau.compute_tau_stats("t", "tau.fits")
        cat = rho.catalogs.get_cat("psf_t")
        return cat.patch_centers, cat.patch, rho.rho_stats, tau.tau_stats

    centers_a, patch_a, rho_a, tau_a = measure()
    default_top = 6
    centers_b, patch_b, rho_b, tau_b = measure()
    np.testing.assert_allclose(centers_a, centers_b, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(patch_a, patch_b)
    for a, b in [(rho_a, rho_b), (tau_a, tau_b)]:
        for name in a.dtype.names:
            np.testing.assert_allclose(a[name], b[name], rtol=1e-10, atol=0)
