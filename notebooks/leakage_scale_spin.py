# ---
# jupyter:
#   jupytext:
#     cell_metadata_json: true
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: shear-psf-leakage-cHl61fDi-py3.9
#     language: python
#     name: python3
# ---

# # Scale-dependent PSF leakage
#
# ## Spin-consistent calculations
#
# Computes and plots scale-dependent spin-consistent PSF leakage functions.
#
# Martin Kilbinger <martin.kilbinger@cea.fr>

# %reload_ext autoreload
# %autoreload 2

# +
import os
import sys
import matplotlib.pylab as plt
from astropy import units
from uncertainties import unumpy

from cs_util import plots as cs_plots
from cs_util import args

import shear_psf_leakage.run_scale as run
from shear_psf_leakage.leakage import *
# -

# ## Compute leakage

# Create leakage instance
obj = run.LeakageScale()


# ### Set input parameters

# Read python parameter file or get user input
params_upd = args.read_param_script("params_leakage_scale.py", obj._params, verbose=True)
for key in params_upd:
    obj._params[key] = params_upd[key]

# ### Run

# Check parameter validity
obj.check_params()

# Prepare output directory and stats file
obj.prepare_output()

# Prepare input
obj.read_data()

# #### Compute correlation function and alpha matrices
# The following command calls `treecorr` to compute auto- and cross-correlation functions.
# This can take a few minutes.

obj.compute_corr_gp_pp_alpha_matrix()

obj.alpha_matrix()

#### For comparison: scalar alpha leakage
obj.compute_corr_gp_pp_alpha()
obj.do_alpha()

# #### PSF auto-correlation function correlation matrix
#
# $$
# r = \frac{ \left( \xi_{12}^\textrm{p,p} \right)^2 }
#     { \xi_{11}^\textrm{p,p} \, \xi_{22}^\textrm{p,p} }
#   = \frac{ \left( \rho_{12, 0} \right)^2 }
#       { \rho_{11, 0} \, \rho_{22, 0} }
# $$

# Check symmetry of PSF auto-correlation matrix
diff = obj.Xi_pp_m[0][1] - obj.Xi_pp_m[1][0]
print(
    "Is r symmetrical? max abs (rel) diff ="
    +f" {max(np.abs(diff)):.3e} ({max(np.abs(diff / obj.Xi_pp_m[0][1])):.3e})",
)

# +
# Plot r and ratios of r

theta = obj.get_theta()

# Exact calculation
r = []
r_ratio_1 = []
r_ratio_2 = []
for ndx in range(len(theta)):
    my_r = (
        obj.Xi_pp_ufloat[ndx][0, 1] ** 2
        / (obj.Xi_pp_ufloat[ndx][0, 0] * obj.Xi_pp_ufloat[ndx][1, 1])
    )
    r.append(my_r)
    r_ratio_1.append(1 / (1 - my_r))
    r_ratio_2.append(my_r / (1 - my_r)) 

print("min max mean r = ", np.min(r), np.max(r), np.mean(r))

# Approximate
r_fast = obj.xi_pp_m[0][1] ** 2 / (obj.xi_pp_m[0][0] * obj.xi_pp_m[1][1])

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

fac = 1.1
xlim = [
    obj._params["theta_min_amin"] / fac,
    obj._params["theta_max_amin"] * fac
]
ylim = (-0.5, 2)

out_path = f"{obj._params['output_dir']}/r.png"

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

theta = obj.get_theta()

xlim = [obj._params["theta_min_amin"] / fac, obj._params["theta_max_amin"] * fac]
ylim = obj._params["leakage_alpha_ylim"]

theta_arr = [theta] * 4
    
alpha = []
yerr = []
labels = []
for idx in (0, 1):
    for jdx in (0, 1):   
        alpha_ufloat = obj.get_alpha_ufloat(idx, jdx)
        alpha.append(unumpy.nominal_values(alpha_ufloat))
        yerr.append(unumpy.std_devs(alpha_ufloat))
        labels.append(rf"$\alpha_{{{idx+1}{jdx+1}}}$")

colors = ["blue", "orange", "orange", "green"]
linestyles = ["-", "-", "--", "-"]
markers = ["o", "^", "v", "s"]

xlabel = r"$\theta$ [arcmin]"
ylabel = r"$\alpha_{ij}(\theta)$"
title = ""
out_path = f"{obj._params['output_dir']}/alpha_ij.png"

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
# Plot spin elements: Spin-0 and spin-4

x0 = 0.5 * (
     obj.get_alpha_ufloat(0, 0)
    + obj.get_alpha_ufloat(1, 1)
)
x4 = 0.5 * (
    obj.get_alpha_ufloat(0, 0)
    - obj.get_alpha_ufloat(1, 1)
)
y4 = 0.5 * (obj.get_alpha_ufloat(0, 1) + obj.get_alpha_ufloat(1, 0))
y0 = 0.5 * (obj.get_alpha_ufloat(1, 0) - obj.get_alpha_ufloat(0, 1))


ylim = obj._params["leakage_alpha_ylim"]

y_arr = [
    unumpy.nominal_values(x0),
    unumpy.nominal_values(x4),
    unumpy.nominal_values(y4),
    unumpy.nominal_values(y0),
]
dy_arr = [
    unumpy.std_devs(x0),
    unumpy.std_devs(x4),
    unumpy.std_devs(y4),
    unumpy.std_devs(y0),
]
theta_arr = [obj.get_theta()] * len(y_arr)
labels = [
    r"$\alpha^\Re_0 = (\alpha_{11} + \alpha_{22})/2$",
    r"$\alpha^\Re_4 = (\alpha_{11} - \alpha_{22})/2$",
    r"$\alpha^\Im_4 = (\alpha_{12} + \alpha_{21})/2$",
    r"$\alpha^\Im_0 = (-\alpha_{12} + \alpha_{21})/2$"
]
colors = ["blue", "orange", "green", "magenta"]
markers = ["o", "s", "h", "^"]
linestyles = ["-"] * 4 

xlabel = r"$\theta$ [arcmin]"
ylabel = r"Components of leakage matrix"
title = ""
out_path = f"{obj._params['output_dir']}/alpha_leakage_m_s0_s4.png"

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
y_arr.append(obj.alpha_leak)
dy_arr.append(obj.sig_alpha_leak)
labels.append(r"$\alpha$ (scalar approx.)")
colors.append("blue")
markers.append("p")
linestyles.append("--")
out_path = f"{obj._params['output_dir']}/alpha_leakage_m_s0_s4_as.png"

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

# #### Consistency relations for scalar leakage

# If the leakage is a scalar function, it can be expressed in three different ways.

# +
xlim = [
    obj._params["theta_min_amin"] / fac,
    obj._params["theta_max_amin"] * fac
]
ylim = obj._params["leakage_alpha_ylim"]

alpha_1 = []
alpha_2 = []
for ndx in range(len(theta)):
    my_a1 = obj.Xi_gp_ufloat[ndx][0, 0] / obj.Xi_pp_ufloat[ndx][0, 0]
    alpha_1.append(my_a1)
    my_a2 = obj.Xi_gp_ufloat[ndx][1, 1] / obj.Xi_pp_ufloat[ndx][1, 1]
    alpha_2.append(my_a2)

y = [
    unumpy.nominal_values(x0),
    obj.alpha_leak,
    unumpy.nominal_values(alpha_1),
    unumpy.nominal_values(alpha_2),
]
dy = [
    unumpy.std_devs(x0),
    obj.sig_alpha_leak,
    unumpy.std_devs(alpha_1),
    unumpy.std_devs(alpha_2),
]
theta_arr = [obj.get_theta()] * len(y)


labels = [r"$\alpha_0^\Re$", r"$\alpha_+$", r"$\alpha_1$", r"$\alpha_2$"]
xlabel = r"$\theta$ [arcmin]"
ylabel = r"$\alpha(\theta)$"
linestyles = ["-", "--", "--", "--"] 
title = ""
out_path = f"{obj._params['output_dir']}/alpha_leakage_scalar_consistency.png"

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
    linestyles=linestyles,
    shift_x=True,
)
# -

# If alpha is a scalar, the mixed-component centered cross-correlation functions should be identical.

# +
xlim = [obj._params["theta_min_amin"], obj._params["theta_max_amin"]]

theta_arr = [theta] * 3

Xi_12 = []
Xi_21 = []
Xi_tr = []
for ndx in range(len(theta)):
    Xi_12.append(obj.Xi_gp_ufloat[ndx][0, 1])
    Xi_21.append(obj.Xi_gp_ufloat[ndx][1, 0])
    Xi_tr.append(obj.Xi_gp_ufloat[ndx][0, 0] + obj.Xi_gp_ufloat[ndx][0, 0])

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

labels = [r"$\tau_{12,0}$", r"$\tau_{21,0}$", r"tr$\tau_0$"]
xlabel = r"$\theta$ [arcmin]"
ylabel = r"correlation functions $\tau_{ij,0}$"
title = ""
out_path = f"{obj._params['output_dir']}/Xi_mixed_consistency.png"

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
)

# +
# For comparison, plot the same for the PSF - PSF correlations

xlim = [obj._params["theta_min_amin"], obj._params["theta_max_amin"]]

theta_arr = [theta] * 3

Xi_12 = []
Xi_21 = []
Xi_tr = []
for ndx in range(len(theta)):
    Xi_12.append(obj.Xi_pp_ufloat[ndx][0, 1])
    Xi_21.append(obj.Xi_pp_ufloat[ndx][1, 0])
    Xi_tr.append(obj.Xi_pp_ufloat[ndx][0, 0] + obj.Xi_pp_ufloat[ndx][0, 0])

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


labels = [r"$\rho_{12,0}$", r"$\rho_{21,0}$", r"tr$\rho_0$"]
xlabel = r"$\theta$ [arcmin]"
ylabel = r"Centered correlation functions"
title = ""
out_path = f"{obj._params['output_dir']}/Xi_pp_mixed_consistency.png"

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
)
