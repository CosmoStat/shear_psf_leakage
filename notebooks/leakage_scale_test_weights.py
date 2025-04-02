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
# ## Testing weighted versus unweighted estimators
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

# +
# Create leakage instances

# Using galaxy weights
obj = run.LeakageScale()

# No weights
obj_nw = run.LeakageScale()

objects = {"weighted": obj, "unweighted": obj_nw}
# -


# ### Set input parameters

# +
params_in_path = "params_leakage_scale_test_weights.py"
if os.path.exists(params_in_path):
    print(f"Reading configuration script {params_in_path}")

    with open(params_in_path) as f:
        exec(f.read())

    # Set instance parameters, copy from above
    for key in params_in:
        obj._params[key] = params_in[key]
        
else:
    raise IOError("configuration script {params_in_path} not found")
# -

for key in obj._params:
    obj_nw._params[key] = obj._params[key]
obj_nw._params["w_col"] = None

print(obj._params)
print(obj_nw._params)

# ### Set up data

# Check parameter validity
for obj in objects.values():
    obj.check_params()

# Prepare output directory and stats file
for obj in objects.values():
    obj.prepare_output()

# Prepare input
for obj in objects.values():
    obj.read_data()

# +
theta = []
alpha_theta = []
yerr = []
labels = []

nx = 6
fx = 1.025

# +
# Compute various correlation functions.
for obj in objects.values():
    obj.compute_corr_gp_pp_alpha()

idx = 0

for key, obj in zip(objects.keys(), objects.values()):
    obj.do_alpha()
    
    theta.append(obj.r_corr_gp.meanr * fx ** (idx - nx))
    alpha_theta.append(obj.alpha_leak)
    yerr.append(obj.sig_alpha_leak)
    labels.append(f"{key} <e^g> != 0")
    idx += 1

# +
# Subtract mean ellipticity

# Weighted
obj = objects["weighted"]
e1_gal = obj.dat_shear[obj._params["e1_col"]]
e2_gal = obj.dat_shear[obj._params["e2_col"]]
w = obj.dat_shear[obj._params["w_col"]]
e1_avg = np.average(e1_gal, weights=w)
e2_avg = np.average(e2_gal, weights=w)
obj.dat_shear[obj._params["e1_col"]] -= e1_avg
obj.dat_shear[obj._params["e2_col"]] -= e2_avg

# Unweighted
obj_nw = objects["unweighted"]
e1_gal = obj_nw.dat_shear[obj_nw._params["e1_col"]]
e2_gal = obj_nw.dat_shear[obj_nw._params["e2_col"]]
e1_avg = np.average(e1_gal)
e2_avg = np.average(e2_gal)
obj_nw.dat_shear[obj_nw._params["e1_col"]] -= e1_avg
obj_nw.dat_shear[obj_nw._params["e2_col"]] -= e2_avg
# -

# Compute various correlation functions.
for obj in objects.values():
    obj.compute_corr_gp_pp_alpha()

# #### Scale-dependent alpha function

# +
# Compute leakage
for key, obj in zip(objects.keys(), objects.values()):
    obj.do_alpha()

    theta.append(obj.r_corr_gp.meanr * fx ** (idx - nx))
    alpha_theta.append(obj.alpha_leak)
    yerr.append(obj.sig_alpha_leak)
    labels.append(key)
    idx += 1

    
# Compute approximate leakage (fast)
for key, obj in zip(objects.keys(), objects.values()):
    obj.do_alpha(fast=True)

    theta.append(obj.r_corr_gp.meanr  * fx ** (idx - nx))
    alpha_theta.append(obj.alpha_leak)
    yerr.append(obj.sig_alpha_leak)
    labels.append(f"{key} approx")
    idx += 1

# +
# Plot

markers = ["o", "s", "d", "p", "v", "*"] 

obj = objects["weighted"]
xlim = [obj._params["theta_min_amin"], obj._params["theta_max_amin"]]
ylim = obj._params["leakage_alpha_ylim"]    
    
xlabel = r"$\theta$ [arcmin]"
ylabel = r"$\alpha(\theta)$"
title = ""
out_path = f"{obj._params['output_dir']}/alpha_leakage_test_weights.png"

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
    markers=markers,
    close_fig=False,
)
