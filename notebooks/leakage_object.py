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
# Demonstration notebook of shear_psf_leakage.run_object.LeakageObject() class.
#
# Martin Kilbinger <martin.kilbinger@cea.fr>

# +
import os
import matplotlib.pylab as plt

from cs_util import canfar

import shear_psf_leakage.run_object as run

# -

# ## Set input parameters

params_in = {}

# ### Paths

# +
# Input galaxy shear catalogue
params_in["input_path_shear"] = "unions_shapepipe_extended_2022_W3_v1.0.3.fits"

# Output directory
params_in["output_dir"] = "leakage_object"
# -

# ### Job control

# +
# Compute (spin-preserving) PSF ellipticity leakage
params_in["PSF_leakage"] = True

# Compute leakage with other parameters
params_in["obs_leakage"] = True

# Other input parameters
params_in["cols"] = "RA Dec e1_PSF fwhm_PSF w mag snr"

# Ratio between two input columns
params_in["cols_ratio"] = "mag_snr"
# -

# ### Other parameters

# +
# PSF ellipticity column names
params_in["e1_PSF_col"] = "e1_PSF"
params_in["e2_PSF_col"] = "e2_PSF"

# Set verbose output
params_in["verbose"] = True
# -

# ### Retrieve test catalogue from VOspace if not yet downloade

vos_dir = "vos:cfis/XXXX/"
canfar.download(
    f"{vos_dir}/{params_in['input_path_shear']}",
    params_in["input_path_shear"],
    verbose=params_in["verbose"],
)

# ## Compute leakage

# Create leakage instance
obj = run.LeakageObject()

# Set instance parameters, copy from above
for key in params_in:
    obj._params[key] = params_in[key]

# ### Run
# There are two options to run the leakage compuations:
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
    obj.PSF_leakage()

if obj._params["obs_leakage"]:
    # Object-by-object spin-consistent PSF leakage
    obj.obs_leakage()
