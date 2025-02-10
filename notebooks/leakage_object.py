# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # PSF leakage: object-wise estimation
#
# Demonstration notebook of shear_psf_leakage.run_object.LeakageObject class.
#
# Martin Kilbinger <martin.kilbinger@cea.fr>

# +
import matplotlib
matplotlib.use("agg")

import os, sys
import matplotlib.pylab as plt

from cs_util import canfar
from cs_util import args

import shear_psf_leakage.run_object as run
from shear_psf_leakage import leakage


# ## Compute leakage

# Create leakage instance
obj = run.LeakageObject()

# Set instance parameters, copy from above
params_upd = args.read_param_script("params_object.py", obj._params, verbose=True)
for key in params_upd:
    obj._params[key] = params_upd[key]

#for key in obj._params:
    #print(key, obj._params[key])

# +
#vos_dir = f"vos:cfis/weak_lensing/DataReleases/v1.0/ShapePipe/{patch}"
#canfar.download(
    #f"{vos_dir}/{obj_params['input_path_shear']}",
    #obj._params["input_path_shear"],
    #verbose=obj._params["verbose"],
#)
# -

# ### Run
# There are two options to run the leakage computations:
# 1. Run all at once with single class routine.
# 2. Execute individual steps.

# ### Option 1. Run all at once

# obj.run()

# ### Option 2. Execute individual steps
# Run commands as given in LeakageObject.run()

# +
# Check parameter validity
obj.check_params()

# Update parameters (here: strings to list)
obj.update_params()

# Prepare output directory
obj.prepare_output()
# -

# Read input catalogue
obj.read_data()

if obj._params["PSF_leakage"]:
    # Object-by-object spin-consistent PSF leakage
    for order in ("lin", "quad"):
            obj.PSF_leakage(mix=True, order=order)

if obj._params["obs_leakage"]:
    # Object-by-object spin-consistent PSF leakage
    obj.obs_leakage()
