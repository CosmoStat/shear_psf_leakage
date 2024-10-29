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

# # PSF contamination: Star-galaxy correlation function $\xi_{\rm{sys}}$
#
# Martin Kilbinger <martin.kilbinger@cea.fr>

# The PSF contamination diagnostic function $\xi_{\rm{sys}} was introduced in Bacon et al. (2003) in the scalar PSF approximation.
# It is defined as
# \begin{equation}
#     \xi_{\textrm{sys}, +}(\theta)
#         = \frac{\left( \xi_+^\textrm{e, p} \right)^2(\theta)}
#                 {\xi_+^\textrm{p, p}(\theta)}
#         = \frac{\tau_{+, 0}^2(\theta)}{\rho_{0, +}(\theta)}
#         = \alpha^2(\theta) \rho_{0, +}(\theta).
# \end{equation}
#
# The generalisation of this function to spin-consistent notation is the matrix
# \begin{equation}
#     \xi_{\textrm{sys}, ij}(\theta)
#     = \sum_{k,l=1}^2 \alpha_{ik} \alpha_{jl} \, \rho_{0, kl}(\theta).
# \end{equation}
# The ‘+’ components can be computed as
# \begin{align}
#     \xi_{\textrm{sys}, +}(\theta)
#     = & \sum_{k,l=1}^2 \sum_{i=1}^2 \alpha_{ik} \alpha_{il} \rho_{0, kl}(\theta)
#     \nonumber \\
#     = & \,
#         \left(
#             \alpha_{11}^2 + \alpha_{21}^2
#         \right) \rho_{0, 11}(\theta)
#         + \left(
#             \alpha_{12}^2 + \alpha_{22}
#         \right)^2 \rho_{0, 22}(\theta)
#         \nonumber \\
#         & \,
#         + 2 \left(
#             \alpha_{11} \alpha_{12} + \alpha_{21} \alpha_{22}
#         \right) \rho_{0, 12}(\theta)
#         \nonumber \\
#     = & \,
#         \left| \alpha_0 + \alpha_4 \right|^2 \rho_{0, 11}(\theta)
#         + \left| \alpha_0 - \alpha_4 \right|^2 \rho_{0, 22}(\theta)
#         \nonumber \\
#         & \,
#         + 4 \left( \alpha_0^\Re \alpha_4^\Im + \alpha_4^\Re \alpha_0^\Im \right)
#         \rho_{0, 12}(\theta)
#         \nonumber \\
#     = & \,
#         2 \left[
#             \left(\alpha_0^\Re \right)^2 + \left(\alpha_4^\Im\right)^2
#             + \left(\alpha_0^\Re \right)^2 + \left(\alpha_4^\Im\right)^2
#         \right]
#         \rho_{0, +}(\theta)
#         \nonumber \\
#       &
#       + 2 \left( \alpha_0^\Re \alpha_4^\Re + \alpha_0^\Im \alpha_4^\Im \right)
#         \left[ \rho_{0, 11}(\theta) - \rho_{0, 22}(\theta) \right]
#        \nonumber \\
#       &
#       + 4 \left( \alpha_0^\Re \alpha_4^\Im + \alpha_4^\Re \alpha_0^\Im \right)
#         \rho_{0, 12}(\theta)
#         ,
# \end{align}

# %reload_ext autoreload
# %autoreload 2
# %matplotlib inline

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

# ## Set up

# Create instance of scale-dependent leakage object
obj_scale = run_scale.LeakageScale()

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

# ### Read input catalogues

# Read input galaxy and star catalogue
obj_scale.read_data()

# #### Compute correlation function and alpha matrices                           
# The following command calls `treecorr` to compute auto- and cross-correlation functions.
# This can take a few minutes.                                                   

obj_scale.compute_corr_gp_pp_alpha_matrix()                                            

obj_scale.alpha_matrix()                                                               

# +
# Compute spin coefficients from matrix elements

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

print(alpha_0_r[0], alpha_0_i[0], alpha_4_r[0], alpha_4_i[0])


# -

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


# +
# xi_sys terms


# Split into +, '-', and mixed parts

xi_sys_term_p = (
    (alpha_0_r ** 2 + alpha_0_i ** 2 + alpha_4_r ** 2 + alpha_4_i ** 2)
    * (get_rho_0(obj_scale, 0, 0) + get_rho_0(obj_scale, 1, 1))
)
xi_sys_term_m = (
    2 * (alpha_0_r * alpha_4_r + alpha_0_i * alpha_4_i)
     * (get_rho_0(obj_scale, 0, 0) - get_rho_0(obj_scale, 1, 1))
)
xi_sys_term_mixed = (
    4 * (
        alpha_0_r * alpha_4_i - alpha_4_r * alpha_0_i
    ) * get_rho_0(obj_scale, 0, 1)
)

xi_sys_tot = xi_sys_term_p + xi_sys_term_m + xi_sys_term_mixed
# -


#### For comparison: scalar xi sys.
obj_scale.compute_corr_gp_pp_alpha()
obj_scale.do_alpha()
obj_scale.compute_xi_sys()


# ## Plotting

# angular scales in arcmin
theta_arcmin = obj_scale.get_theta()

# +
# Mean ellipticities for centered correlation functions

e1_gal = obj_scale.dat_shear["e1"]
e2_gal = obj_scale.dat_shear["e2"]
weights_gal = obj_scale.dat_shear["w"]

complex_gal = (                                                         
    np.average(e1_gal, weights=weights_gal)                             
    + np.average(e2_gal, weights=weights_gal) * 1j
)

e1_star = obj_scale.dat_PSF["E1_STAR_HSM"]
e2_star = obj_scale.dat_PSF["E2_STAR_HSM"]
complex_psf = np.mean(e1_star) + np.mean(e2_star) * 1j

# +
# At the moment we compute the scalar xi_sys here using centered correlation functions.
# TODO: Implement consistent centering in shear_psf_leakage classes. 

# xi_sys = tau_0^2/rho_0
xi_sys_scalar = (obj_scale.r_corr_gp.xip - np.real(np.conj(complex_gal) * complex_psf)) ** 2 / (obj_scale.r_corr_pp.xip - np.abs((complex_psf) ** 2))
std_xi_sys_scalar = obj_scale.C_sys_std_p

y = [
    unumpy.nominal_values(xi_sys_term_p),
    unumpy.nominal_values(xi_sys_term_m),
    unumpy.nominal_values(xi_sys_term_mixed),
    unumpy.nominal_values(xi_sys_tot),
    xi_sys_scalar,
    
]
dy = [
    unumpy.std_devs(xi_sys_term_p),
    unumpy.std_devs(xi_sys_term_m),
    unumpy.std_devs(xi_sys_term_mixed),
    unumpy.std_devs(xi_sys_tot),
    std_xi_sys_scalar,
]
x = [theta_arcmin] * len(y)

title = r"Bacon et al. (2003) $\xi_{sys}$"
xlabel = r"$\theta$ [arcmin]" 
ylabel = "terms"
out_path = f"{obj_scale._params['output_dir']}/xi_sys_terms.png"                                  
labels = ["$t_+$", "$t_-$", r"$t_{\rm mixed}$", r"$\sum t_i$", "scalar"]
markers = ["o", "d", "^", "x", "s"]

ylim = [-1e-6, 2e-6]

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
    ylim=ylim,
    markers=markers,
)
# +
# Consistency test. Different computations of xi_sys terms

# Using a_ij matrix
xi_sys_term_m_11 = (
    (obj_scale.get_alpha_ufloat(0, 0) ** 2 + obj_scale.get_alpha_ufloat(1, 0) ** 2)
    * get_rho_0(obj_scale, 0, 0)
)
xi_sys_term_m_22 = (
    (obj_scale.get_alpha_ufloat(0, 1) ** 2 + obj_scale.get_alpha_ufloat(1, 1) ** 2)
    * get_rho_0(obj_scale, 1, 1)
)
xi_sys_term_m_12 = (
    2 * (obj_scale.get_alpha_ufloat(0, 0) * obj_scale.get_alpha_ufloat(0, 1)
     + obj_scale.get_alpha_ufloat(1, 0) * obj_scale.get_alpha_ufloat(1, 1))
     * get_rho_0(obj_scale, 0, 1)
)

# Sum of the three terms
xi_sys_m_tot =  xi_sys_term_m_11 + xi_sys_term_m_22 + xi_sys_term_m_12

# Matrix sum
xi_sys_m2_tot = 0
for idx in (0, 1):
    for kdx in (0, 1):
        for ldx in (0, 1):
            xi_sys_m2_tot +=  (
                obj_scale.get_alpha_ufloat(idx, kdx)
                * obj_scale.get_alpha_ufloat(idx, ldx)
                * get_rho_0(obj_scale, kdx, ldx)
            )

y = [
    unumpy.nominal_values(xi_sys_term_m_11),
    unumpy.nominal_values(xi_sys_term_m_22),
    unumpy.nominal_values(xi_sys_term_m_12),
    unumpy.nominal_values(xi_sys_m_tot),
    unumpy.nominal_values(xi_sys_m2_tot),
    unumpy.nominal_values(xi_sys_tot),
    
]
dy = [
    unumpy.std_devs(xi_sys_term_m_11),
    unumpy.std_devs(xi_sys_term_m_22),
    unumpy.std_devs(xi_sys_term_m_12),
    unumpy.std_devs(xi_sys_m_tot),
    unumpy.std_devs(xi_sys_m2_tot),
    unumpy.std_devs(xi_sys_tot),
]
x = [theta_arcmin] * len(y)

title = r"Bacon et al. (2003) $\xi_{sys}$"
xlabel = r"$\theta$ [arcmin]" 
ylabel = "terms with matrix coeffs"
out_path = f"{obj_scale._params['output_dir']}/xi_sys_terms_m.png"                                  
labels = ["$t_{11}$", "$t_{22}$", r"$t_{12}$", r"$\sum t_i$ (matrix terms)", r"$\sum t_i$ (matrix terms 2)", "$\sum t_i$ (spin terms)"]
markers = ["o", "d", "^", "x", "h", "s"]
linestyles = [":"] * 3
linestyles.extend(["-", "-.", "--"])

ylim = [-1e-6, 2e-6]

cs_plots.plot_data_1d(
    x,
    y,
    dy,
    title,
    xlabel,
    ylabel,
    out_path,
    labels=labels,
    shift_x=False,
    xlog=True,
    ylog=False,
    ylim=ylim,
    markers=markers,
    linestyles=linestyles,
)



# +
# Plot difference, should be consistent with zero

d1 = xi_sys_m_tot - xi_sys_tot
d2 = xi_sys_m2_tot - xi_sys_tot
d3 = xi_sys_m_tot - xi_sys_m2_tot

d4 = xi_sys_term_m_11 + xi_sys_term_m_22 - (xi_sys_term_p + xi_sys_term_m)
d5 = xi_sys_term_m_12 - xi_sys_term_mixed

y = [
    unumpy.nominal_values(d1),
    unumpy.nominal_values(d2),
    unumpy.nominal_values(d3),
    unumpy.nominal_values(d4),
    unumpy.nominal_values(d5),
]
dy = [
    unumpy.std_devs(d1),
    unumpy.std_devs(d2),
    unumpy.std_devs(d3),
    #unumpy.std_devs(d4),
    #unumpy.std_devs(d5),
]
x = [theta_arcmin] * len(y)

title = r"Bacon et al. (2003) $\xi_{sys}$"
xlabel = r"$\theta$ [arcmin]" 
ylabel = "difference"
out_path = f"{obj_scale._params['output_dir']}/xi_sys_diff.png"                                  
labels = [r"(matrix - spin) tot", r"(matrix 2 - spin) tot", "(matrix - matrix 2) tot", "11 22", "mixed"]
markers = ["o", "d", "s", "x", "v"]
linestyles = ["-", "--", ":", ":", ":"]


cs_plots.plot_data_1d(
    x[:3],
    y[:3],
    dy[:3],
    title,
    xlabel,
    ylabel,
    out_path,
    labels=labels,
    shift_x=True,
    xlog=True,
    ylog=False,
    markers=markers,
    linestyles=linestyles,
)

# +
# Test: compare different estimates of xi_sys. Some use uncentered, some centered correlation functions.

plt.clf()
plt.loglog(theta_arcmin, obj_scale.C_sys_p, "p:", label="scalar uncentered")

# spin t_+
plt.loglog(
    theta_arcmin,
    unumpy.nominal_values(xi_sys_term_p), 
    "d-",
    label="spin $t_+$ centered",
)

# tau^2 / rho
plt.loglog(theta_arcmin, obj_scale.r_corr_gp.xip ** 2 / obj_scale.r_corr_pp.xip, "v:", label="scalar centered")
plt.loglog(
    theta_arcmin,
    (obj_scale.r_corr_gp.xip - np.real(np.conj(complex_gal) * complex_psf)) ** 2 / (obj_scale.r_corr_pp.xip - np.abs((complex_psf) ** 2)),
    "v-",
    label="3c"
)

plt.legend()
plt.show()
plt.savefig(f"{obj_scale._params['output_dir']}/xi_sys_test.png")

# -


