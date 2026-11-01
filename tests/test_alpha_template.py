"""Object-wise leakage must propagate through the template like galaxy shear."""

import numpy as np
import pytest
import treecorr
from shear_psf_leakage.rho_tau_stat import TauStat


@pytest.mark.parametrize("use_eta", [False, True])
def test_alpha_template_matches_objectwise_injection(use_eta):
    rng = np.random.default_rng(61)
    n = 600
    names = [
        "RA",
        "Dec",
        "w",
        "e1",
        "e2",
        "p1",
        "p2",
        "sp1",
        "sp2",
        "s1",
        "s2",
        "tp",
        "ts",
    ]
    data = np.zeros(n, dtype=[(name, "f8") for name in names])
    data["RA"] = rng.uniform(10, 13, n)
    data["Dec"] = rng.uniform(-2, 1, n)
    data["w"] = rng.uniform(0.2, 3, n)
    for i in (1, 2):
        # Large nonzero means and spatial structure expose missing centring.
        data[f"p{i}"] = (
            0.03 * i + 0.01 * np.sin(data["RA"] * i) + rng.normal(0, 0.005, n)
        )
        data[f"sp{i}"] = data[f"p{i}"] + rng.normal(0, 0.003, n)
        data[f"s{i}"] = data[f"sp{i}"] + rng.normal(0, 0.002, n)
    data["tp"] = 0.8
    data["ts"] = rng.uniform(0.85, 1.1, n)
    params = dict(
        ra_col="RA",
        dec_col="Dec",
        w_col="w",
        e1_col="e1",
        e2_col="e2",
        e1_gal_PSF_col="p1",
        e2_gal_PSF_col="p2",
        ra_PSF_col="RA",
        dec_PSF_col="Dec",
        e1_PSF_col="sp1",
        e2_PSF_col="sp2",
        e1_star_col="s1",
        e2_star_col="s2",
        PSF_size="tp",
        star_size="ts",
        ra_units="deg",
        dec_units="deg",
        patch_number=8,
        R11=0.7,
        R22=1.2,
    )
    cfg = dict(
        min_sep=2,
        max_sep=180,
        nbins=5,
        sep_units="arcmin",
        bin_slop=0,
        var_method="jackknife",
        cross_patch_weight="match",
        num_threads=1,
    )
    tau = TauStat(
        params=params, treecorr_config=cfg, use_eta=use_eta, verbose=False
    )
    # Distinct star and galaxy samples and a mask applied after centring.
    tau.build_cat_to_compute_tau(data[::3], "psf", "test")
    mask = data["RA"] > 10.5
    tau.build_cat_to_compute_tau(data, "gal_psf", "test", mask=mask)
    template = tau.compute_alpha_template("test")
    assert len(template) == (3 if use_eta else 2)
    alpha = 0.1
    data["e1"] = alpha * data["p1"]
    data["e2"] = alpha * data["p2"]
    tau.build_cat_to_compute_tau(data, "gal", "test", mask=mask)
    fields = ["psf", "psf_error"] + (["psf_size_error"] if use_eta else [])
    measured = []
    for field, predictor in zip(fields, template):
        corr = treecorr.GGCorrelation(cfg)
        corr.process(
            tau.catalogs.get_cat("gal_test"),
            tau.catalogs.get_cat(field + "_test"),
        )
        np.testing.assert_allclose(
            corr.xip, alpha * predictor.xip, rtol=1e-6, atol=2e-12
        )
        np.testing.assert_allclose(
            corr.xim, alpha * predictor.xim, rtol=1e-6, atol=2e-12
        )
        measured.append(corr)

    def values(corrs):
        return np.concatenate([corr.xip for corr in corrs])

    rows, _ = treecorr.build_multi_cov_design_matrix(
        measured, method="jackknife", func=values, cross_patch_weight="match"
    )
    predictors, _ = treecorr.build_multi_cov_design_matrix(
        template, method="jackknife", func=values, cross_patch_weight="match"
    )
    np.testing.assert_allclose(rows, alpha * predictors, rtol=1e-6, atol=2e-12)
    # This assertion rejects the original star-sampled, uncentred rho column.
    rho = treecorr.GGCorrelation(cfg)
    rho.process(tau.catalogs.get_cat("psf_test"))
    assert not np.allclose(
        measured[0].xip, alpha * rho.xip, rtol=0.01, atol=1e-12
    )
