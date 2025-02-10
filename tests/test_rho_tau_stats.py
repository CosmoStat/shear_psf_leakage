import sys
sys.path.append('../shear_psf_leakage')

import numpy as np
import matplotlib.pyplot as plt
from shear_psf_leakage.rho_tau_stat import RhoStat, TauStat, PSFErrorFit
from shear_psf_leakage.plots import plot_contours


#Run tests on shapepipe V1 and DESY3

#First specify  the paths to your data
paths_gal = [
    #"/n17data/mkilbing/astro/data/CFIS/v1.0/ShapePipe/unions_shapepipe_extended_2022_v1.0.fits",
    #"/automnt/n23data1/fhervas/shapepipe_clean_uppercut/unions_shapepipe_extended_2022_v1.0.4.fits",
    "/home/mkilbing/astro/data/DES/DES_Y3_cut.fits",
    "/n17data/mkilbing/astro/data/CFIS/v1.0/SP_LFmask/unions_shapepipe_extended_2022_v1.0_mtheli8k.fits",
    "/n17data/mkilbing/astro/data/CFIS/v1.0/SP_LFmask/unions_shapepipe_extended_2022_v1.3_mtheli8k.fits",
    "/n17data/mkilbing/astro/data/CFIS/v0.0/shapepipe_1500_goldshape_v1.fits"
]

paths_psf = [
    #"/n17data/mkilbing/astro/data/CFIS/v1.0/ShapePipe/unions_shapepipe_psf_2022_v1.0.2.fits",
    "/home/mkilbing/astro/data/DES/psf_y3a1-v29.fits",
    "/n17data/mkilbing/astro/data/CFIS/v1.0/SP_LFmask/unions_shapepipe_psf_2022_v1.0.2_mtheli8k.fits",
    "/n17data/mkilbing/astro/data/CFIS/v1.0/SP_LFmask/unions_shapepipe_psf_2022_v1.0.2_mtheli8k.fits",
    "/n17data/sguerrini/star_cat.fits"
]

#Specify if you want to mask some data using flags
masks = [
    #True,
    False,
    True,
    True,
    True
]

#Specify is the sizes have to be squared. It can depend on the catalog.
square_sizes = [
    #True,
    False,
    True,
    True,
    True
]

output = '/home/guerrini/rho_tau_stats_output/new_setup_2'

#Contains the params of your catalog. Don't forget to specify them since they can vary between different catalogs
params_sp = {
    "e1_col": "e1",
    "e2_col": "e2",
    "w_col": "w",
    "ra_col": "RA",
    "dec_col": "Dec",
    "e1_PSF_col": "E1_PSF_HSM",
    "e2_PSF_col": "E2_PSF_HSM",
    "e1_star_col": "E1_STAR_HSM",
    "e2_star_col": "E2_STAR_HSM",
    "PSF_size": "SIGMA_PSF_HSM",
    "star_size": "SIGMA_STAR_HSM",
    "PSF_flag": "FLAG_PSF_HSM",
    "star_flag": "FLAG_STAR_HSM",
    "patch_number": 200,
    "ra_units": "deg",
    "dec_units": "deg" 
}


params_des = {
    "e1_col": "e1_cal",
    "e2_col": "e2_cal",
    "w_col": "w",
    "ra_col": "RA",
    "dec_col": "Dec",
    "e1_PSF_col": "piff_e1",
    "e2_PSF_col": "piff_e2",
    "e1_star_col": "obs_e1",
    "e2_star_col": "obs_e2",
    "PSF_size": "piff_T",
    "star_size": "obs_T",
    "patch_number": 120,
    "ra_units": "deg",
    "dec_units": "deg" 
}

params_axel = {
    "e1_col": "g1",
    "e2_col": "g2",
    "w_col": "w",
    "ra_col": "RA",
    "dec_col": "Dec",
    "e1_PSF_col": "E1_PSF_HSM",
    "e2_PSF_col": "E2_PSF_HSM",
    "e1_star_col": "E1_STAR_HSM",
    "e2_star_col": "E2_STAR_HSM",
    "PSF_size": "SIGMA_PSF_HSM",
    "star_size": "SIGMA_STAR_HSM",
    "PSF_flag": "FLAG_PSF_HSM",
    "star_flag": "FLAG_STAR_HSM",
    "patch_number": 120,
    "ra_units": "deg",
    "dec_units": "deg" 
}

treecorr_config = {
                "ra_units": "deg",
                "dec_units": "deg",
                "sep_units": "arcmin",
                "min_sep": 0.1,
                "max_sep": 250,
                "nbins": 20,
                "var_method": "jackknife"
}

params = [
    params_des,
    params_sp,
    params_sp,
    params_axel
]

colors = ['red', 'blue', 'green', 'purple'] #Colors for the plot
catalog_ids = ['DES', 'SPV1.0', 'SPV1.3', 'SP1500'] #Ids of the catalogs

print(paths_gal)
print(paths_psf)
###################################Compute, save and plot the rho statistics############################
rho_stat_handler = RhoStat(output=output, treecorr_config=treecorr_config, verbose=True) #Create your class to compute, save, load and plot rho_stats
for path_gal, path_psf, cat_id, param, mask, square_size in zip(paths_gal, paths_psf, catalog_ids, params, masks, square_sizes): #Iterate on the different catalogs
    try:
        rho_stat_handler.load_rho_stats('rho_stats_'+cat_id+'.fits')
    except FileNotFoundError:
        if param is None:
            rho_stat_handler.catalogs.params_default(output)
        else:
            rho_stat_handler.catalogs.set_params(param, output) #Set the right parameters
        rho_stat_handler.build_cat_to_compute_rho(path_psf, catalog_id=cat_id, square_size=square_size, mask=mask) #Build the different catalogs
        rho_stat_handler.compute_rho_stats(cat_id, 'rho_stats_'+cat_id+'.fits') #Compute and save the rho statistics

filenames = ['rho_stats_'+cat_id+'.fits' for cat_id in catalog_ids]

rho_stat_handler.plot_rho_stats(filenames, colors, catalog_ids, abs=True, savefig='rho_stats.png') #Plot

######################################Compute, save and plot the tau statistics#############################
tau_stat_handler = TauStat(catalogs=rho_stat_handler.catalogs, output=output, treecorr_config=treecorr_config, verbose=True) #Create your class to compute, save, load and plot tau_stats

for path_gal, path_psf, cat_id, param, mask, square_size in zip(paths_gal, paths_psf, catalog_ids, params, masks, square_sizes): #Iterate on the catalogs

    try:
        tau_stat_handler.load_tau_stats('tau_stats_'+cat_id+'.fits')
    except FileNotFoundError:
        if param is None:
            tau_stat_handler.catalogs.params_default(output)
        else:
            tau_stat_handler.catalogs.set_params(param, output) #Set the parameters

        if 'psf_'+cat_id not in tau_stat_handler.catalogs.catalogs_dict.keys():
            tau_stat_handler.build_cat_to_compute_tau(path_psf, cat_type='psf', catalog_id=cat_id, square_size=square_size, mask=mask) #Build the catalog of galaxies. PSF was computed above
        tau_stat_handler.build_cat_to_compute_tau(path_gal, cat_type='gal', catalog_id=cat_id, square_size=square_size, mask=mask) #Build the catalog of galaxies. PSF was computed above
        


        only_p = lambda corrs: np.array([corr.xip for corr in corrs]).flatten() #function to extract the tau+
        tau_stat_handler.compute_tau_stats(cat_id, 'tau_stats_'+cat_id+'.fits', save_cov=True, func=only_p, var_method='jackknife') #Compute and save the tau statistics


filenames = ['tau_stats_'+cat_id+'.fits' for cat_id in catalog_ids]
tau_stat_handler.plot_tau_stats(filenames, colors, catalog_ids, savefig='tau_stats.png', plot_tau_m=False) #Plot

#######################################Fit the error model to rho and tau statistics data##########################

psf_fitter = PSFErrorFit(rho_stat_handler, tau_stat_handler, output)

flat_sample_list = []
mcmc_result_list = []
q_list = []

for cat_id, param in zip(catalog_ids, params):
    psf_fitter.load_rho_stat('rho_stats_'+cat_id+'.fits')
    psf_fitter.load_tau_stat('tau_stats_'+cat_id+'.fits')
    psf_fitter.load_covariance('cov_'+cat_id+'.npy')

    flat_samples, mcmc_result, q = psf_fitter.run_chain(savefig='mcmc_samples_'+cat_id+'.png', npatch=param["patch_number"], apply_debias=True)
    flat_sample_list.append(flat_samples)
    mcmc_result_list.append(mcmc_result)
    q_list.append(q)

    psf_fitter.plot_tau_stats_w_model(mcmc_result[1], 'tau_stats_'+cat_id+'.fits', 'blue', cat_id, savefig='best_fit_'+cat_id+'.png')


plot_contours(flat_sample_list, names=['x0', 'x1', 'x2'], labels=['alpha', 'beta', 'eta'], savefig=output+'/contours_tau_stat.png',
    legend_labels=catalog_ids, legend_loc='upper right', contour_colors=colors, markers={'x0':0, 'x1':1, 'x2':1}
)

plt.figure(figsize=(15, 6))

for mcmc_result, cat_id, color, flat_sample in zip(mcmc_result_list, catalog_ids, colors, flat_sample_list):
    psf_fitter.load_rho_stat('rho_stats_'+cat_id+'.fits')
    for i in range(100):
        psf_fitter.plot_xi_psf_sys(flat_sample[-i+1], cat_id, color, alpha=0.1)
    psf_fitter.plot_xi_psf_sys(mcmc_result[1], cat_id, color)
plt.legend()
plt.savefig(output+'/xi_sys.png')
