# ---
# jupyter:
#   jupytext:
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

# # PSF contamination: PSF leakage diagnostic
#
# Martin Kilbinger <martin.kilbinger@cea.fr>

# %reload_ext autoreload
# %autoreload 2

# +
import os
import matplotlib.pylab as plt
import numpy as np

from astropy import units

from uncertainties import ufloat
from uncertainties import unumpy                                                 

from cs_util import canfar
from cs_util import plots as cs_plots
from cs_util import args

from shear_psf_leakage import run_object
from shear_psf_leakage import run_scale
from shear_psf_leakage import leakage
# -

# ## Compute leakage

# Create leakage instance
obj_scale = run_scale.LeakageScale()

# ## Set input parameters

# Read python parameter file or get user input
params_upd = args.read_param_script("params_xi_sys.py", obj_scale._params, verbose=True)
for key in params_upd:
    obj_scale._params[key] = params_upd[key]


# +
# Check parameter validity
obj_scale.check_params()

# Prepare output directory
obj_scale.prepare_output()
# -

# Read input catalogue
obj_scale.read_data()

# #### Compute correlation function and alpha matrices                           
# The following command calls `treecorr` to compute auto- and cross-correlation functions.
# This can take a few minutes.                                                   

obj_scale.compute_corr_gp_pp_alpha_matrix()                                            

obj_scale.alpha_matrix()                                                               

# Spin leakage elements
alpha_0_r = (
    0.5 * (
        obj_scale.get_alpha_ufloat(0, 0)
        + obj_scale.get_alpha_ufloat(1, 1)
    )
)
alpha_0_i = (
    0.5 * (
        -obj_scale.get_alpha_ufloat(0, 1)
        + obj_scale.get_alpha_ufloat(1, 0)
    )
)
alpha_4_r = (
    0.5 * (
        obj_scale.get_alpha_ufloat(0, 0)
        - obj_scale.get_alpha_ufloat(1, 1)
    )
)
alpha_4_i = (
    0.5 * (
        obj_scale.get_alpha_ufloat(0, 1)
        + obj_scale.get_alpha_ufloat(1, 0)
    )
)


def get_rho_0(self, idx, jdx):                                        
	"""Get Rho 0.                                                     

	Return (i, j) matrix element of rho_0.                

	Parameters                                                               
	----------                                                               
	idx : int                                                                
		line index, allowed are 0 or 1                                       
	jdx : int                                                                
		column index, allowed are 0 or 1                                     

	Returns                                                                  
	-------                                                                  
	numpy.ndarray                                                            
		matrix element as array over scales, each entry is                   
		of type ufloat                                                       

	"""                                                                      
	mat = []                                                                 
	n_theta = self._params["n_theta"]                                        
	for ndx in range(n_theta):                                               
		mat.append(self.Xi_pp_ufloat[ndx][idx, jdx])
	return np.array(mat)


# xi_sys terms
print("xi_sys terms")
xi_sys_term_p = (
    2 * (
        alpha_0_r ** 2 + alpha_0_i ** 2 + alpha_4_r ** 2 + alpha_4_i ** 2
    ) * (get_rho_0(obj_scale, 0, 0) + get_rho_0(obj_scale, 1, 1))
)
xi_sys_term_m = 2 * alpha_0_i * alpha_4_i * (
    (get_rho_0(obj_scale, 0, 0) - get_rho_0(obj_scale, 1, 1))
)
xi_sys_term_mixed = (
    4 * (
        alpha_0_r * alpha_4_i - alpha_4_r * alpha_0_i
    ) * get_rho_0(obj_scale, 0, 1)
)
print("xi_sys done")


#### For comparison: scalar xi sys
obj_scale.compute_corr_gp_pp_alpha()
obj_scale.do_alpha()
obj_scale.compute_xi_sys()


# Plot terms
theta_arcmin = obj_scale.get_theta()

# +
y = [
    unumpy.nominal_values(xi_sys_term_p),
    unumpy.nominal_values(xi_sys_term_m),
    unumpy.nominal_values(xi_sys_term_mixed),
    obj_scale.C_sys_p,
]
dy = [
    unumpy.std_devs(xi_sys_term_p),
    unumpy.std_devs(xi_sys_term_m),
    unumpy.std_devs(xi_sys_term_mixed),
    obj_scale.C_sys_std_p,
]
x = [theta_arcmin] * len(y)

title = r"Bacon et al. (2003) $\xi_{sys}$"
xlabel = r"$\theta$ [arcmin]" 
ylabel = "terms"
out_path = f"{obj_scale._params['output_dir']}/xi_sys_terms.png"                                  
labels = ["$t_+$", "$t_-$", r"$t_{\rm mixed}$", "scalar"]

cs_plots.plot_data_1d(
    x,
    y,
    dy,
    title,
    xlabel,
    ylabel,
    out_path,
    labels=labels,
    shift_x=True,
    xlog=True,
    ylog=False,
)
# -
obj_scale.C_sys_p

print(unumpy.nominal_values(
    obj_scale.alpha_leak ** 2 * (
        get_rho_0(obj_scale, 0, 0) + get_rho_0(obj_scale, 1, 1)
    )
))
print()
print(obj_scale.alpha_leak ** 2 * obj_scale.r_corr_pp.xip)
print()
print(obj_scale.C_sys_p)

obj_scale.alpha_leak

alpha_4_i


