# ---
# jupyter:
#   jupytext:
#     cell_metadata_json: true
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.1
#   kernelspec:
#     display_name: sp_validation
#     language: python
#     name: python3
# ---

# # PSF contamination: Scale-dependent PSF leakage
#
# ## Spin-consistent formalism
#
# Martin Kilbinger <martin.kilbinger@cea.fr>

# %reload_ext autoreload
# %autoreload 2
# %matplotlib inline

# +
import os
import sys
import matplotlib.pylab as plt
from astropy import units
from uncertainties import unumpy

from cs_util import plots as cs_plots
from cs_util import args

from shear_psf_leakage import run_scale
from shear_psf_leakage.leakage import *
# -

# ## Set up

# Create instance of scale-dependent leakage object
obj_scale = run_scale.LeakageScale()


# ### Set input parameters

# Read python parameter file or get user input
params_upd = args.read_param_script(
    "params_leakage_scale.py",
    obj_scale._params,
    verbose=True
)
for key in params_upd:
    obj_scale._params[key] = params_upd[key]

# ### Run

# +
# Check parameter validity
obj_scale.check_params()

# Prepare output directory and stats file
obj_scale.prepare_output()
# -

# ### Read input catalogues

# Read input galaxy and star catalogue
obj_scale.read_data()

# #### Compute correlation function and alpha matrices
# The following command calls `treecorr` to compute auto- and cross-correlation functions.
# This can take a few minutes.

obj_scale.compute_corr_gp_pp_alpha_matrix()

obj_scale.alpha_matrix()

#### For comparison: scalar alpha leakage
obj_scale.compute_corr_gp_pp_alpha()
obj_scale.do_alpha()

# #### PSF auto-correlation function correlation matrix
#
# $$
# r = \frac{ \left( \xi_{12}^\textrm{p,p} \right)^2 }
#     { \xi_{11}^\textrm{p,p} \, \xi_{22}^\textrm{p,p} }
#   = \frac{ \left( \rho_{12, 0} \right)^2 }
#       { \rho_{11, 0} \, \rho_{22, 0} }
# $$

# Check symmetry of PSF auto-correlation matrix
diff = obj_scale.Xi_pp_m[0][1] - obj_scale.Xi_pp_m[1][0]
print(
    "Is r symmetrical? max abs (rel) diff ="
    + f" {max(np.abs(diff)):.3e}"
    + f" ({max(np.abs(diff / obj_scale.Xi_pp_m[0][1])):.3e})",
)

# +
# Plot r and ratios of r

theta = obj_scale.get_theta()

# Exact: Using centered correlation functions
r = []
r_ratio_1 = []
r_ratio_2 = []
for ndx in range(len(theta)):
    my_r = (
        obj_scale.Xi_pp_ufloat[ndx][0, 1] ** 2
        / (obj_scale.Xi_pp_ufloat[ndx][0, 0] * obj_scale.Xi_pp_ufloat[ndx][1, 1])
    )
    r.append(my_r)
    r_ratio_1.append(1 / (1 - my_r))
    r_ratio_2.append(my_r / (1 - my_r)) 

print("min max mean r = ", np.min(r), np.max(r), np.mean(r))

# Approximate: Using uncentered correlation functions
r_fast = obj_scale.xi_pp_m[0][1] ** 2 / (obj_scale.xi_pp_m[0][0] * obj_scale.xi_pp_m[1][1])

n = 6
theta_arr = [theta] * n
r_arr = []
dr_arr = []
    
r_arr.append(unumpy.nominal_values(r))
r_arr.append(unumpy.nominal_values(r_ratio_1))
r_arr.append(unumpy.nominal_values(r_ratio_2))
r_arr.append(r_fast)
r_arr.append(1 / (1 - r_fast))
r_arr.append(r_fast / (1 - r_fast))

dr_arr.append(unumpy.std_devs(r))
dr_arr.append(unumpy.std_devs(r_ratio_1))
dr_arr.append(unumpy.std_devs(r_ratio_2))
for idx in range(3):
    dr_arr.append(np.nan)

labels = ["$r$", "$1/(1-r)$", "$r/(1-r)$", "", "", ""]
colors = ["blue", "orange", "green", "blue", "orange", "green"]
linestyles = ["-"] * 3 + ["--"] * 3
linewidths = [2] * 3 + [1] * 3

xlabel = r"$\theta$ [arcmin]"
ylabel = r"functions of $r(\theta)$"

fac = 0.9
xlim = [
    obj_scale._params["theta_min_amin"] * fac,
    obj_scale._params["theta_max_amin"],
]
ylim = (-0.5, 2)

out_path = f"{obj_scale._params['output_dir']}/r.png"

title = ""

cs_plots.plot_data_1d(
    theta_arr,
    r_arr,
    dr_arr,
    title,
    xlabel,
    ylabel,
    out_path,
    labels=labels,
    xlog=True,
    xlim=xlim,
    ylim=ylim,
    colors=colors,
    linewidths=linewidths,
    linestyles=linestyles,
    close_fig=False,
    shift_x=True,
)

# +
# Plot alpha matrix elements

ylim = obj_scale._params["leakage_alpha_ylim"]

n = 4
theta_arr = [theta] * n
    
alpha = []
yerr = []
labels = []
for idx in (0, 1):
    for jdx in (0, 1):   
        alpha_ufloat = obj_scale.get_alpha_ufloat(idx, jdx)
        alpha.append(unumpy.nominal_values(alpha_ufloat))
        yerr.append(unumpy.std_devs(alpha_ufloat))
        labels.append(rf"$\alpha_{{{idx+1}{jdx+1}}}$")

colors = ["blue", "orange", "orange", "green"]
linestyles = ["-", "-", "--", "-"]
markers = ["o", "^", "v", "s"]

xlabel = r"$\theta$ [arcmin]"
ylabel = r"$\alpha_{ij}(\theta)$"
title = ""
out_path = f"{obj_scale._params['output_dir']}/alpha_ij.png"

cs_plots.plot_data_1d(
    theta_arr,
    alpha,
    yerr,
    title,
    xlabel,
    ylabel,
    out_path,
    labels=labels,
    xlog=True,
    xlim=xlim,
    ylim=ylim,
    colors=colors,
    linestyles=linestyles,
    markers=markers,
    close_fig=False,
    shift_x=True,
)

# +
# Plot spin coefficients

obj_scale.compute_alpha_spin_coeffs()

n = 4
theta_arr = [theta] * n

y_arr = [
    unumpy.nominal_values(obj_scale._alpha_0_r),
    unumpy.nominal_values(obj_scale._alpha_0_i),
    unumpy.nominal_values(obj_scale._alpha_4_r),
    unumpy.nominal_values(obj_scale._alpha_4_i),
]
dy_arr = [
    unumpy.std_devs(obj_scale._alpha_0_r),
    unumpy.std_devs(obj_scale._alpha_0_i),
    unumpy.std_devs(obj_scale._alpha_4_r),
    unumpy.std_devs(obj_scale._alpha_4_i),
]
labels = [
    r"$\alpha^\Re_0$",
    r"$\alpha^\Im_0$",
    r"$\alpha^\Re_r$",
    r"$\alpha^\Im_r$"
]
colors = ["blue", "orange", "green", "magenta"]
markers = ["o", "s", "^", "v"]
linestyles = ["-"] * 4 

xlabel = r"$\theta$ [arcmin]"
ylabel = r"Components of leakage matrix"
title = ""
out_path = f"{obj_scale._params['output_dir']}/alpha_leakage_m_s0_s4.png"

cs_plots.plot_data_1d(
    theta_arr,
    y_arr,
    dy_arr,
    title,
    xlabel,
    ylabel,
    out_path,
    labels=labels,
    xlog=True,
    xlim=xlim,
    ylim=ylim,
    colors=colors,
    markers=markers,
    close_fig=False,
    shift_x=True,
)

# Including scalar leakage for comparison
theta_arr.append(theta)
y_arr.append(obj_scale.alpha_leak)
dy_arr.append(obj_scale.sig_alpha_leak)
labels.append(r"$\alpha^{\rm s}_{+}$")
colors.append("cyan")
markers.append("x")
linestyles.append("--")
out_path = f"{obj_scale._params['output_dir']}/alpha_leakage_m_s0_s4_as.png"

cs_plots.plot_data_1d(
    theta_arr,
    y_arr,
    dy_arr,
    title,
    xlabel,
    ylabel,
    out_path,
    labels=labels,
    xlog=True,
    xlim=xlim,
    ylim=ylim,
    colors=colors,
    markers=markers,
    linestyles=linestyles,
    close_fig=False,
    shift_x=True,
)
# + \xi_ {"incorrectly_encoded_metadata": "{22}^\\textrm{g,p} \\, \\xi_{11}^\\textrm{p,p}"}
# #### Consistency relations for scalar leakage

# If the leakage is a scalar function, it can be expressed in three different ways.

# +
alpha_1 = []
alpha_2 = []
for ndx in range(len(theta)):
    my_a1 = obj_scale.Xi_gp_ufloat[ndx][0, 0] / obj_scale.Xi_pp_ufloat[ndx][0, 0]
    alpha_1.append(my_a1)
    my_a2 = obj_scale.Xi_gp_ufloat[ndx][1, 1] / obj_scale.Xi_pp_ufloat[ndx][1, 1]
    alpha_2.append(my_a2)

# TODO: Use centered functions for all cases

y = [
    obj_scale.alpha_leak,
    unumpy.nominal_values(alpha_1),
    unumpy.nominal_values(alpha_2),
    unumpy.nominal_values(obj_scale._alpha_0_r),
]
dy = [
    obj_scale.sig_alpha_leak,
    unumpy.std_devs(alpha_1),
    unumpy.std_devs(alpha_2),
    unumpy.std_devs(obj_scale._alpha_0_r),
]
theta_arr = [theta] * len(y)

labels = [
    r"$\alpha^{\rm s}_+$",
    r"$\alpha^{\rm s}_1$",
    r"$\alpha^{\rm s}_2$",
    r"$\alpha^\Re_0$",
]
xlabel = r"$\theta$ [arcmin]"
ylabel = r"$\alpha(\theta)$"
title = ""
out_path = f"{obj_scale._params['output_dir']}/alpha_leakage_scalar_consistency.png"
colors = ["cyan", "black", "grey", "blue"]
markers = ["x", "h", "o", "o"]
linestyles = ["--", "--", "--", "-"]

cs_plots.plot_data_1d(
    theta_arr,
    y,
    dy,
    title,
    xlabel,
    ylabel,
    out_path,
    labels=labels,
    xlog=True,
    xlim=xlim,
    ylim=ylim,
    close_fig=False,
    shift_x=True,
    colors=colors,
    markers=markers,
    linestyles=linestyles,
)
# -

# If alpha is a scalar, the mixed-component centered cross-correlation functions should be identical.

# +
Xi_12 = []
Xi_21 = []
Xi_tr = []
for ndx in range(len(theta)):
    Xi_12.append(obj_scale.Xi_gp_ufloat[ndx][0, 1])
    Xi_21.append(obj_scale.Xi_gp_ufloat[ndx][1, 0])
    Xi_tr.append(obj_scale.Xi_gp_ufloat[ndx][0, 0] + obj_scale.Xi_gp_ufloat[ndx][0, 0])

y = [
    unumpy.nominal_values(Xi_12),
    unumpy.nominal_values(Xi_21),
    unumpy.nominal_values(Xi_tr),
]
dy = [
    unumpy.std_devs(Xi_12),
    unumpy.std_devs(Xi_21),
    unumpy.std_devs(Xi_tr),
]
theta_arr = [theta] * len(y)

labels = [r"$\tau_{0, 12}$", r"$\tau_{0,21}$", r"tr$\tau_{0}$"]
xlabel = r"$\theta$ [arcmin]"
ylabel = r"correlation functions $\tau_{0,ij}$"
title = ""
out_path = f"{obj_scale._params['output_dir']}/Xi_mixed_consistency.png"
markers = ["o", "s", "d"]
linestyles = ["-", "--", ":"]

cs_plots.plot_data_1d(
    theta_arr,
    y,
    dy,
    title,
    xlabel,
    ylabel,
    out_path,
    labels=labels,
    xlog=True,
    xlim=xlim,
    close_fig=False,
    shift_x=True,
    markers=markers,
    linestyles=linestyles,
)

# +
# For comparison, plot the same for the PSF - PSF correlations

Xi_12 = []
Xi_21 = []
Xi_tr = []
for ndx in range(len(theta)):
    Xi_12.append(obj_scale.Xi_pp_ufloat[ndx][0, 1])
    Xi_21.append(obj_scale.Xi_pp_ufloat[ndx][1, 0])
    Xi_tr.append(obj_scale.Xi_pp_ufloat[ndx][0, 0] + obj_scale.Xi_pp_ufloat[ndx][0, 0])

y = [
    unumpy.nominal_values(Xi_12),
    unumpy.nominal_values(Xi_21),
    unumpy.nominal_values(Xi_tr),
]
dy = [
    unumpy.std_devs(Xi_12),
    unumpy.std_devs(Xi_21),
    unumpy.std_devs(Xi_tr),
]
theta_arr = [theta] * len(y)

labels = [r"$\rho_{0,12}$", r"$\rho_{0,21}$", r"tr$\rho_0$"]
xlabel = r"$\theta$ [arcmin]"
ylabel = r"Centered correlation functions"
title = ""
out_path = f"{obj_scale._params['output_dir']}/Xi_pp_mixed_consistency.png"
markers = ["o", "s", "d"]
linestyles = ["-", "--", ":"]

cs_plots.plot_data_1d(
    theta_arr,
    y,
    dy,
    title,
    xlabel,
    ylabel,
    out_path,
    labels=labels,
    xlog=True,
    xlim=xlim,
    close_fig=False,
    shift_x=True,
    markers=markers,
    linestyles=linestyles,
)
# -


