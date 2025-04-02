# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: shear_psf_leakage
#     language: python
#     name: shear_psf_leakage
# ---

# # Scale-dependent PSF leakage
#
# ## Scalar $\alpha(\theta)$ and $\xi_\textrm{sys}$
#
# package shear_psf_leakge
#
# Martin Kilbinger <martin.kilbinger@cea.fr>

# %reload_ext autoreload
# %autoreload 2

# +
import os
import sys
import matplotlib.pylab as plt
from astropy import units

from cs_util import plots as cs_plots

import shear_psf_leakage.run_scale as run
from shear_psf_leakage.leakage import *
# -

# ## Compute leakage

# Create leakage instance
obj = run.LeakageScale()


# ### Set input parameters

params_in_path = "params_leakage_scale.py"
if os.path.exists(params_in_path):
    print(f"Reading configuration script {params_in_path}")

    with open(params_in_path) as f:
        exec(f.read())

    # Set instance parameters, copy from above
    for key in params_in:
        obj._params[key] = params_in[key]
else:
    print(f"Configuration script {params_in_path} not found, asking for user input")

    for key in obj._params:
        msg = f"{key}? [{obj._params[key]}] "
        val_user = input(msg)
        if val_user != "":
            obj._params[key] = val_user

print(obj._params)

# + [markdown] jp-MarkdownHeadingCollapsed=true
# ### Run
# There are two options to run the leakage compuations:
# 1. Run all at once with single class routine.
# 2. Execute individual steps.
# -

# ### Option 1. Run all at once

# +
# obj.run()
# -

# ### Option 2. Execute individual steps

# Check parameter validity
obj.check_params()

# Prepare output directory and stats file
obj.prepare_output()

# Prepare input
obj.read_data()

# #### Compute correlation functions
# The following command calls `treecorr` to compute auto- and cross-correlation functions.
# This can take a few minutes.

# Compute various correlation functions.
obj.compute_corr_gp_pp_alpha()

# #### Scale-dependent alpha function

# Compute correct leakage
obj.do_alpha()

# +
# Plot

xlim = [obj._params["theta_min_amin"], obj._params["theta_max_amin"]]
ylim = obj._params["leakage_alpha_ylim"]

theta = [obj.r_corr_gp.meanr]
alpha_theta = [obj.alpha_leak]
yerr = [obj.sig_alpha_leak]
labels = [r"$\alpha$"]
xlabel = r"$\theta$ [arcmin]"
ylabel = r"$\alpha(\theta)$"
title = ""
out_path = f"{obj._params['output_dir']}/alpha_leakage.png"

cs_plots.plot_data_1d(
    theta,
    alpha_theta,
    yerr,
    title,
    xlabel,
    ylabel,
    out_path,
    labels=labels,
    xlog=True,
    xlim=xlim,
    ylim=ylim,
    close_fig=False,
)
# -

# #### xi_sys cross-correlation function

# Compute
obj.do_xi_sys()

# Plot
obj.plot_xi_sys(close_fig=False)

# Theoretical model for xi_pm
xi_p_theo, xi_m_theo = run.get_theo_xi(
    obj.r_corr_gp.meanr * units.arcmin,
    obj._params["dndz_path"],
)

# Plot ratio
obj.plot_xi_sys_ratio(xi_p_theo, xi_m_theo, close_fig=False)
