# Shear PSF leakage

## About

This library quantifies the PSF leakage on weak-lensing galaxy shear
measurements. It computes the following quantities:

- The object-wise PSF leakage at linear and quadratic order, using regression.
- The object-wise dependency of galaxy shear to other (scalar) observables.
- The scale-dependent PSF leakage function galaxy - PSF leakage using
  two-point correlation functions.

## Installation

The fastest way to Install this library is with `pip` automatically from
pypi.org,

```bash
pip install shear_psf_leakage
```

Alternatively, you can use TBD

## Authors

Martin Kilbinger <martin.kilbinger@cea.fr>

Clara Bonini

Sacha Guerrini

Axel Guinot

### Galaxy-position alpha template

For object-wise leakage `e_gal += alpha * p_gal`, the alpha predictor of tau
is the PSF at **galaxy positions**, processed with the same galaxy weights,
weighted mean subtraction, response and selection as the galaxy ellipticity.
A star-sampled rho column can differ through both its mean and its sampling.

Supply the two galaxy PSF column names in the usual catalogue parameters;
these are separate from `e1_PSF_col` and `e2_PSF_col`, which describe the stars:

```python
params.update(e1_gal_PSF_col="e1_PSF", e2_gal_PSF_col="e2_PSF")
tau = TauStat(params=params, treecorr_config=treecorr_config)
tau.build_cat_to_compute_tau(stars, "psf", catalog_id="sample")
tau.build_cat_to_compute_tau(galaxies, "gal", catalog_id="sample", mask=mask)
tau.build_cat_to_compute_tau(galaxies, "gal_psf", catalog_id="sample", mask=mask)
alpha_correlations = tau.compute_alpha_template("sample")
alpha_column = np.concatenate([corr.xip for corr in alpha_correlations])
```

Here `params` is the same parameter dictionary used for the galaxy and star
catalogues; `np` is NumPy, and `TauStat` is imported from
`shear_psf_leakage.rho_tau_stat`.
The returned TreeCorr correlations follow tau0, tau2, tau5 order (without
tau5 when `use_eta=False`).
Use `alpha_column` **instead of** the rho alpha column in the fit design.
This measurement API does not change `PSFErrorFit`'s rho-based model or its
samplers; a consumer must explicitly use the measured column in its design.
For errors-in-variables fitting, build the alpha-template and tau delete-one
rows jointly using the same patches and cross-patch weights; retain their
cross-covariance.
The returned objects preserve the patch results for this purpose.
Use the same full galaxy table, response values, selection mask and stars
for both measurements.
Centring precedes the selection mask, following the `gal` convention.
The rho statistics remain the PSF auto-correlations used to predict
contamination auto-power; do not replace those products with this template.
This corrects the alpha template only; beta and eta still use the existing
star-sampled predictors.
