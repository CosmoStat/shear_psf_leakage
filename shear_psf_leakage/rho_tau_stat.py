"""
This module sets up a class to compute the rho stats computation

Author: Sacha Guerrini
"""

from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table

import treecorr
import emcee


def neg_dash(
    ax,
    x_in,
    y_in,
    yerr_in,
    vertical_lines=True,
    xlabel='',
    ylabel='',
    rho_nb='',
    tau_nb='',
    cat_id='',
    ylim=None,
    semilogx=False,
    semilogy=False,
    **kwargs
):
    r"""Neg Dash.

    This function is for making plots with vertical errorbars,
    where negative values are shown in absolute value as dashed lines.
    The resulting plot can either be saved by specifying a file name as
    ``plot_name``, or be kept as a pyplot instance (for instance to combine
    several neg dashes).

    Parameters
    ----------
    ax :
        The matplotlib object on which the plot is performed
    x_in : numpy.ndarray
        X-axis inputs
    y_in : numpy.ndarray
        Y-axis inputs
    yerr_in : numpy.ndarray
        Y-axis error inputs
    vertical_lines : bool, optional
        Option to plot vertical lines; default is ``True``
    xlabel : str, optional
        X-axis label
    ylabel : str, optional
        Y-axis label
    rho_nb : str, optional
        Rho number
    ylim : float, optional
        Y-axis limit
    semilogx : bool
        Option to plot the x-axis in log scale; default is ``False``
    semilogy : bool
        Option to plot the y-axis in log scale; default is ``False``

    """
    x = np.copy(x_in)
    y = np.copy(y_in)
    if yerr_in is not None:
        yerr = np.copy(yerr_in)
    else:
        yerr = np.zeros_like(x)
    # catch and separate errorbar-specific keywords from Lines2D ones
    safekwargs = dict(kwargs)
    errbkwargs = dict()
    if 'linestyle' in kwargs.keys():
        print(
            'Warning: linestyle was provided but that would kind of defeat'
            + 'the purpose, so I will just ignore it. Sorry.'
        )
        del safekwargs['linestyle']
    for errorbar_kword in [
        'fmt', 'ecolor', 'elinewidth', 'capsize', 'barsabove', 'errorevery'
    ]:
        if errorbar_kword in kwargs.keys():
            # posfmt = '-'+kwargs['fmt']
            # negfmt = '--'+kwargs['fmt']
            errbkwargs[errorbar_kword] = kwargs[errorbar_kword]
            del safekwargs[errorbar_kword]
    errbkwargs = dict(errbkwargs, **safekwargs)

    # plot up to next change of sign
    current_sign = np.sign(y[0])
    first_change = np.argmax(current_sign * y < 0)
    while first_change:
        if current_sign > 0:
            ax.errorbar(
                x[:first_change],
                y[:first_change],
                yerr=yerr[:first_change],
                linestyle='-',
                **errbkwargs,
            )
            if vertical_lines:
                ax.vlines(
                    x[first_change - 1],
                    0,
                    y[first_change - 1],
                    linestyle='-',
                    **safekwargs,
                )
                ax.vlines(
                    x[first_change],
                    0,
                    np.abs(y[first_change]),
                    linestyle='--',
                    **safekwargs,
                )
        else:
            ax.errorbar(
                x[:first_change],
                np.abs(y[:first_change]),
                yerr=yerr[:first_change],
                linestyle='--',
                **errbkwargs,
            )
            if vertical_lines:
                ax.vlines(
                    x[first_change - 1],
                    0,
                    np.abs(y[first_change - 1]),
                    linestyle='--',
                    **safekwargs,
                )
                ax.vlines(
                    x[first_change],
                    0,
                    y[first_change],
                    linestyle='-',
                    **safekwargs,
                )
        x = x[first_change:]
        y = y[first_change:]
        yerr = yerr[first_change:]
        current_sign *= -1
        first_change = np.argmax(current_sign * y < 0)
    # one last time when `first_change'==0 ie no more changes:
    if rho_nb:
        lab = fr'$\rho_{rho_nb}(\theta)$ '+cat_id
    elif tau_nb:
        lab = fr'$\tau_{tau_nb}(\theta)$' +cat_id
    else:
        lab = cat_id
    if current_sign > 0:
        ax.errorbar(x, y, yerr=yerr, linestyle='-', label=lab, **errbkwargs)
    else:
        ax.errorbar(x, np.abs(y), yerr=yerr, linestyle='--', label=lab,
                     **errbkwargs)
    if semilogx:
        ax.set_xscale('log')
    if semilogy:
        ax.set_yscale('log')
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

class Catalogs():
    """
    Catalogs

    Class to build the different treecorr catalogs given a shape catalog that will be
    used to compute the different statistics
    """

    def __init__(self, params=None, output=None):
        #set default parameters
        if (params is None):
            self.params_default(output)
        else:
            self.set_params(params, output)

        self.catalogs_dict = dict()
        self.dat_shear = None
        self.dat_psf = None

    def params_default(self, output=None):
        """
        Params Default.

        Initialize the parameters of the class with columns name from SPV1.
        For the treecorr configuration, default parameters are:
        -coord_units: degree
        -sep_units: arcmin
        -theta_min: 0.1
        -theta_max: 100
        -n_theta: 20
        -var_method: jackknife !!Requires to set a patch number for the different catalogues!!
        """

        self._params = {
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
            "patch_number": 120,
            "ra_units": "deg",
            "dec_units": "deg"
        }

        if output is not None:
            self._output = output
        else:
            self._output="."

    def set_params(self, params=None, output=None):
        """
        set_params:

        Initialize the parameters to the given configuration. Can also update the current params.
        """
        if params is not None:
            self._params = params
        if output is not None:
            self._output = output

    def read_shear_cat(self, path_gal, path_psf, hdu=1):
        """
        read_shear_cat

        Read a shear catalogue with galaxies ('gal') or stars ('psf').
        Only one such catalogue can be loaded at a time.

        Raises
        ------
        AssertionError: Please specify a path for the shear catalog you want to read.
        """
        assert ((path_gal is not None) or (path_psf is not None)), ("Please specify a path for the shear catalog you want to read.")
        if path_gal is not None:
            self.dat_shear = fits.getdata(path_gal, ext=hdu)
        if path_psf is not None:
            self.dat_psf = fits.getdata(path_psf, ext=hdu)

    def get_cat_fields(self, cat_type, square_size=False):
        """
        Get Cat Fields

        Get catalogue fields for correlation.

        Parameters
        ----------
        cat_type : str
            catalogue type, allowed are 'gal', 'psf', 'psf_error' or 'psf_size_error'

        square_size : bool
            If True, the size computed in the catalogue is squared (Default: False)

        Returns
        -------
        np.array
            ra
        np.array
            dec
        np.array
            e1
        np.array
            e2
        np.array
            weights; 'None' is cat_type is not 'gal'. Returns a one_like array if weights are not specified

        Raises
        ------
        AssertionError
            If the specified cat_type does not belong to the allowed list.
        """

        allowed_types = ['gal', 'psf', 'psf_error', 'psf_size_error']

        assert cat_type in allowed_types, ("The specified catalogue type is invalid. Check the one you use is allowed."
                                           "Allowed cat_type: 'gal', 'psf', 'psf_error', 'psf_size_error'.")

        if cat_type=="gal":
            if self._params["w_col"] is not None:
                weights = self.dat_shear[self._params["w_col"]]
            else:
                weights = np.ones_like(ra)
            assert self.dat_shear is not None, ("Check you read the shear catalogs correctly.")
            ra = self.dat_shear[self._params["ra_col"]]
            dec = self.dat_shear[self._params["dec_col"]]
            g1 = self.dat_shear[self._params["e1_col"]] - np.average(self.dat_shear[self._params["e1_col"]], weights=weights)
            g2 = self.dat_shear[self._params["e2_col"]] - np.average(self.dat_shear[self._params["e2_col"]], weights=weights)
            if self._params.get("R11", None) is not None:
                g1 /= self._params["R11"]
            if self._params.get("R22", None) is not None:
                g2 /= self._params["R22"]
        else:
            assert self.dat_psf is not None, ("Check you read the shear catalogs correctly.")
            #Add a mask?
            #mask = (self.dat_psf[self._params["FLAG_PSF_HSM"]]==0) & (self.dat_psf[self._params["FLAG_STAR_HSM"]]==0)
            ra = self.dat_psf[self._params["ra_col"]]
            dec = self.dat_psf[self._params["dec_col"]]
            weights = None

            if cat_type=="psf":
                g1 = self.dat_psf[self._params["e1_PSF_col"]]# - self.dat_psf[self._params["e1_PSF_col"]].mean()
                g2 = self.dat_psf[self._params["e2_PSF_col"]]# - self.dat_psf[self._params["e2_PSF_col"]].mean()

            elif cat_type=="psf_error":
                g1 = (self.dat_psf[self._params["e1_star_col"]] - self.dat_psf[self._params["e1_PSF_col"]])
                #g1 -= g1.mean()
                g2 = (self.dat_psf[self._params["e2_star_col"]] - self.dat_psf[self._params["e2_PSF_col"]])
                #g2 -= g2.mean()

            else:
                size_star = self.dat_psf[self._params["star_size"]]**2 if square_size else  self.dat_psf[self._params["star_size"]]
                size_psf = self.dat_psf[self._params["PSF_size"]]**2 if square_size else  self.dat_psf[self._params["PSF_size"]]

                g1 = self.dat_psf[self._params["e1_star_col"]] * (size_star - size_psf) / size_star
                #g1 -= g1.mean()
                g2 = self.dat_psf[self._params["e2_star_col"]] * (size_star - size_psf) / size_star
                #g2 -= g2.mean()

        return ra, dec, g1, g2, weights
    
    def build_catalog(self, cat_type, key, npatch=None, patch_centers=None, square_size=False, mask=False):
        """
        build_catalogue

        Build a treecorr.Catalog of a certain type using the class _params. A key is given as input
        to identify the catalog in self.catalogs_dict.

        Parameters
        ----------
        cat_type : str
            catalogue type, allowed are 'gal', 'psf', 'psf_error' or 'psf_size_error'.

        key : str
            String used to key the catalog to an entry of the dict of catalogs.

        npatch : int
            number of patch used to compute variance with jackknife or bootstrap. (Default: value in self._params)

        square_size : bool
            If True, the size computed in the catalogue is squared (Default: False)

        mask : bool
            If True, use PSF and star flags to mask the data. (Default: False)
        """

        if npatch is None:
            npatch = self._params["patch_number"]

        ra, dec, g1, g2, weights = self.get_cat_fields(cat_type, square_size)

        if mask:
            flag_psf = self.dat_psf[self._params["PSF_flag"]]
            flag_star = self.dat_psf[self._params["star_flag"]]
            mask_arr = (flag_psf==0) & (flag_star==0)
            if weights is not None:
                weights = weights[mask_arr]
        else:
            mask_arr = np.array([True for i in ra])

        if patch_centers is None:
            cat = treecorr.Catalog(
                ra=ra[mask_arr],
                dec=dec[mask_arr],
                g1=g1[mask_arr],
                g2=g2[mask_arr],
                w=weights,
                ra_units=self._params["ra_units"],
                dec_units=self._params["dec_units"],
                npatch=npatch
            )
        else:
            cat = treecorr.Catalog(
                ra=ra[mask_arr],
                dec=dec[mask_arr],
                g1=g1[mask_arr],
                g2=g2[mask_arr],
                w=weights,
                ra_units=self._params["ra_units"],
                dec_units=self._params["dec_units"],
                patch_centers=patch_centers
            )

        self.catalogs_dict.update(
            {key: cat}
        )

    def delete_catalog(self, key):
        """
        delete_catalog

        Delete the catalog instance mapped by the string key.
        """

        try:
            self.catalogs_dict.pop(key)
        except KeyError:
            print("This entry did not exist.")

    def show_catalogs(self):
        """
        show_catalogs

        Print the keys of the element stored in self.catalogs_dict
        """
        for key in self.catalogs_dict.keys():
            print(key)

    def get_cat(self, key):
        """
        get_cat

        Return the catalogue stored with the given key

        Parameters
        ----------
        key : str
            The key used to identify the catalogue in the dictionary

        Returns
        -------
        treecorr.Catalog
            The requested treecorr.Catalog
        """
        return self.catalogs_dict[key]

class RhoStat():
    """
    RhoStat

    Class to compute the rho statistics (Rowe 2010) of a PSF catalogue.
    """

    def __init__(
        self,
        params=None,
        output=None,
        treecorr_config=None,
        verbose=False
    ):

        self.catalogs = Catalogs(params, output)

        if treecorr_config is None:
            self._treecorr_config = {
                "ra_units": "deg",
                "dec_units": "deg",
                "sep_units": "arcmin",
                "min_sep": 0.1,
                "max_sep": 100,
                "nbins": 20,
                "var_method": "jackknife"
            }
        else:
            self._treecorr_config = treecorr_config

        self.verbose = verbose

    def build_cat_to_compute_rho(self, path_cat_star, catalog_id='', square_size=False, mask=False, hdu=1):
        """
        build_cat_to_compute_rho

        Parameters
        ----------
        path_cat_star : str
            Path to the catalog of stars used to compute the rho-statistics.

        catalog_id : str
            An id to identify the catalog used in the keys of the stored treecorr.Catalog.

        square_size : bool
            If True, the size computed in the catalogue is squared (Default: False)

        mask : bool
            If True, use PSF and star flags to mask the data. (Default: False)

        hdu : int, optional
            HDU number of input FITS file, default is 1
        """

        self.catalogs.read_shear_cat(path_gal=None, path_psf=path_cat_star, hdu=hdu)

        if self.verbose:
            print("Building catalogs...")

        self.catalogs.build_catalog(cat_type='psf', key='psf_'+catalog_id, square_size=square_size, mask=mask)
        patch_centers = self.catalogs.catalogs_dict['psf_'+catalog_id].patch_centers
        self.catalogs.build_catalog(cat_type='psf_error', key='psf_error_'+catalog_id, patch_centers=patch_centers, square_size=square_size, mask=mask)
        self.catalogs.build_catalog(cat_type='psf_size_error', key='psf_size_error_'+catalog_id, patch_centers=patch_centers, square_size=square_size, mask=mask)

        if self.verbose:
            print("Catalogs successfully built...")
            self.catalogs.show_catalogs()

    def compute_rho_stats(self, catalog_id, filename, save_cov=False, func=None, var_method='jackknife'):
        """
        compute_rho_stats

        Compute the rho statistics of your psf. Store it as an attribute of the class which could be save afterwards.

        Parameters:
        ----------
        catalog_id : str
            The id of the catalog used to compute the rho statistics.

        filename : str
            The path where the rho stats will be saved.
        """
        if self.verbose:
            print("Computation of the rho statistics of "+catalog_id+" in progress...")
        rho_0 = treecorr.GGCorrelation(self._treecorr_config)
        rho_0.process(self.catalogs.get_cat('psf_'+catalog_id), self.catalogs.get_cat('psf_'+catalog_id))
        rho_1 = treecorr.GGCorrelation(self._treecorr_config)
        rho_1.process(self.catalogs.get_cat('psf_error_'+catalog_id), self.catalogs.get_cat('psf_error_'+catalog_id))
        rho_2 = treecorr.GGCorrelation(self._treecorr_config)
        rho_2.process(self.catalogs.get_cat('psf_error_'+catalog_id), self.catalogs.get_cat('psf_'+catalog_id))
        rho_3 = treecorr.GGCorrelation(self._treecorr_config)
        rho_3.process(self.catalogs.get_cat('psf_size_error_'+catalog_id), self.catalogs.get_cat('psf_size_error_'+catalog_id))
        rho_4 = treecorr.GGCorrelation(self._treecorr_config)
        rho_4.process(self.catalogs.get_cat('psf_error_'+catalog_id), self.catalogs.get_cat('psf_size_error_'+catalog_id))
        rho_5 = treecorr.GGCorrelation(self._treecorr_config)
        rho_5.process(self.catalogs.get_cat('psf_'+catalog_id), self.catalogs.get_cat('psf_size_error_'+catalog_id))

        self.rho_stats = Table(
            [
                rho_0.meanr,
                rho_0.xip,
                rho_0.varxip,
                rho_0.xim,
                rho_1.varxim,
                rho_1.xip,
                rho_1.varxip,
                rho_1.xim,
                rho_1.varxim,
                rho_2.xip,
                rho_2.varxip,
                rho_2.xim,
                rho_2.varxim,
                rho_3.xip,
                rho_3.varxip,
                rho_3.xim,
                rho_3.varxim,
                rho_4.xip,
                rho_4.varxip,
                rho_4.xim,
                rho_4.varxim,
                rho_5.xip,
                rho_5.varxip,
                rho_5.xim,
                rho_5.varxim,
            ],
            names=(
                'theta',
                'rho_0_p',
                'varrho_0_p',
                'rho_0_m',
                'varrho_0_m',
                'rho_1_p',
                'varrho_1_p',
                'rho_1_m',
                'varrho_1_m',
                'rho_2_p',
                'varrho_2_p',
                'rho_2_m',
                'varrho_2_m',
                'rho_3_p',
                'varrho_3_p',
                'rho_3_m',
                'varrho_3_m',
                'rho_4_p',
                'varrho_4_p',
                'rho_4_m',
                'varrho_4_m',
                'rho_5_p',
                'varrho_5_p',
                'rho_5_m',
                'varrho_5_m',
            )
        )

        if self.verbose:
            print("Done...")

        if save_cov:
            if self.verbose:
                print("Computing the covariance...")

            cov = treecorr.estimate_multi_cov([rho_0, rho_1, rho_2, rho_3, rho_4, rho_5], var_method, func)

            np.save(self.catalogs._output+'/'+'cov_rho_'+catalog_id, cov)

        self.save_rho_stats(filename) #A bit dirty just because of consistency of the datatype :/
        self.load_rho_stats(filename)

    def save_rho_stats(self, filename):
        self.rho_stats.write(self.catalogs._output+'/'+filename, format='fits', overwrite=True)

    def load_rho_stats(self, filename):
        self.rho_stats = fits.getdata(self.catalogs._output+'/'+filename)

    def plot_rho_stats(
        self,
        filenames,
        colors,
        catalog_ids,
        abs=True,
        savefig=None,
        legend="each",
        title=None,
    ):
        """
        plot_rho_stats

        Method to plot Rho + statistics of several catalogues given in argument. Figures are saved in PNG format.

        Parameters:
        ----------
        filenames : list str
            List of the files containing the rho statistics. They can be computed using the method `compute_rho_stats` of this class.

        colors : list str
            Color of the plot for the different catalogs. We recommend using different colors for different catalogs for readability.

        catalogs_id : list str
            A list of catalogs id to label accurately the legend.

        abs : bool
            If True, plot the absolute value of the rho-statistics. Otherwise, plot the negative values with dashed lines.

        savefig : str
            If not None, saves the figure with the name given in savefig.

        legend : str, optional
            allowed are "each" (default; legends in each panel);
            "outside" (legend outside of panels); "none" (no legend)

        title : str, optional
            global plot tite, default is ``None``
        """

        fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(15,9))
        ax = ax.flatten()

        for filename, color, cat_id in zip(filenames, colors, catalog_ids): #Plot for the different catalogs
            self.load_rho_stats(filename)

            for i in range(6):
                xlabel=r"$\theta$ [arcmin]" if i>2 else ''

                if legend == "each":
                    ylabel = r"$\rho-$statistics" if (i==0 or i==3) else ''
                    label = fr'$\rho_{i}(\theta)$ {cat_id}'
                elif legend == "outside":
                    ylabel = rf"$\rho_i(\theta)$"
                    label = fr'$\rho_i$ {cat_id}'

                if abs:
                    ax[i].errorbar(self.rho_stats['theta'], np.abs(self.rho_stats['rho_'+str(i)+'_p']), yerr=np.sqrt(self.rho_stats['varrho_'+str(i)+'_p']),
                    label=label, color=color, capsize=2)
                    ax[i].set_xlabel(xlabel)
                    ax[i].set_ylabel(ylabel)
                    ax[i].set_xscale('log')
                    ax[i].set_yscale('log')
                else:
                    #Plot the negative values of the rho-stats in dashed lines
                    neg_dash(
                        ax[i], self.rho_stats['theta'], self.rho_stats['rho_'+str(i)+'_p'], yerr_in=np.sqrt(self.rho_stats['varrho_'+str(i)+'_p']),
                        vertical_lines=False, rho_nb=str(i), cat_id=cat_id, xlabel=xlabel, ylabel=ylabel, semilogx=True, semilogy=True, capsize=True, color=color,
                    )
                ax[i].set_xlim(float(self._treecorr_config["min_sep"]), float(self._treecorr_config["max_sep"]))

                if legend == "each":
                    ax[i].legend(loc='best', fontsize='small')

        if legend == "outside":
            ax[-1].legend(bbox_to_anchor=(1.5, 0.0), fontsize='small')

        if title:
            plt.suptitle(title)

        plt.tight_layout()
        if savefig is not None:
            plt.savefig(self.catalogs._output+'/'+savefig, bbox_inches='tight')

        plt.close()

class TauStat():
    """
    TauStat

    Class to compute the tau statistics (Gatti 2022) of a PSF and gal catalogue.
    """

    def __init__(self, params=None, output=None, treecorr_config=None, catalogs=None, verbose=False):

        if catalogs is None:
            self.catalogs = Catalogs(params, output)
        else:
            self.catalogs = catalogs

        if treecorr_config is None:
            self._treecorr_config = {
                "ra_units": "deg",
                "dec_units": "deg",
                "sep_units": "arcmin",
                "min_sep": 0.1,
                "max_sep": 100,
                "nbins": 20,
                "var_method": "jackknife"
            }
        else:
            self._treecorr_config = treecorr_config

        self.verbose = verbose

    def build_cat_to_compute_tau(self, path_cat, cat_type, catalog_id='', square_size=False, mask=False, hdu=1):
        """
        build_cat_to_compute_tau

        Parameters
        ----------
        path_cat : str
            Path to the catalog built to compute the tau-statistics.

        cat_type : str
            Specify the type of the catalogue to build. 'gal' or 'psf'

        catalog_id : str
            An id to identify the catalog used in the keys of the stored treecorr.Catalog.

        square_size : bool
            If True, the size computed in the catalogue is squared (Default: False)

        mask : bool
            If True, use PSF and star flags to mask the data. (Default: False)

        hdu : int, optional
            HDU number of input FITS file, default is 1
        """

        if cat_type=="psf":
            self.catalogs.read_shear_cat(path_gal=None, path_psf=path_cat, hdu=hdu)

            if self.verbose:
                print("Building catalogs...")

            self.catalogs.build_catalog(cat_type='psf', key='psf_'+catalog_id, square_size=square_size, mask=mask)
            patch_centers = self.catalogs.catalogs_dict['psf_'+catalog_id].patch_centers
            self.catalogs.build_catalog(cat_type='psf_error', key='psf_error_'+catalog_id, patch_centers=patch_centers, square_size=square_size, mask=mask)
            self.catalogs.build_catalog(cat_type='psf_size_error', key='psf_size_error_'+catalog_id, patch_centers=patch_centers, square_size=square_size, mask=mask)

        else:
            self.catalogs.read_shear_cat(path_gal=path_cat, path_psf=None, hdu=hdu)

            if self.verbose:
                print("Building catalog...")
            
            try:
                patch_centers = self.catalogs.catalogs_dict['psf_'+catalog_id].patch_centers
            except KeyError:
                print("Warning, you should build psf catalog before galaxy catalog.")
                patch_centers = None
            self.catalogs.build_catalog(cat_type='gal', key='gal_'+catalog_id, patch_centers=patch_centers)

        if self.verbose:
            print("Catalogs successfully built...")
            self.catalogs.show_catalogs()

    def compute_tau_stats(self, catalog_id, filename, save_cov=False, func=None, var_method='jackknife'):
        """
        compute_tau_stats

        Compute the tau statistics of your catalog and save it. Can also compute a covariance related to tau statistics and save it.

        Parameters:
        ----------
        catalog_id : str
            The id of the catalog used to compute the rho statistics.

        filename : str
            The path where the rho stats will be saved.

        save_cov : bool
            If True, compute and save a covariance related to tau statistics.

        func : function
            The function to select the quantity whose covariance is being computed from tau_0, tau_2 and tau_5.

        var_method: str
            The method used to compute the covariance. (Default: jackknife)
        """

        if self.verbose:
            print("Computation of the tau statistics of "+catalog_id+" in progress...")
        tau_0 = treecorr.GGCorrelation(self._treecorr_config)
        tau_0.process(self.catalogs.get_cat('gal_'+catalog_id), self.catalogs.get_cat('psf_'+catalog_id))
        tau_2 = treecorr.GGCorrelation(self._treecorr_config)
        tau_2.process(self.catalogs.get_cat('gal_'+catalog_id), self.catalogs.get_cat('psf_error_'+catalog_id))
        tau_5 = treecorr.GGCorrelation(self._treecorr_config)
        tau_5.process(self.catalogs.get_cat('gal_'+catalog_id), self.catalogs.get_cat('psf_size_error_'+catalog_id))

        self.tau_stats = Table(
            [
                tau_0.meanr,
                tau_0.xip,
                tau_0.varxip,
                tau_0.xim,
                tau_0.varxim,
                tau_2.xip,
                tau_2.varxip,
                tau_2.xim,
                tau_2.varxim,
                tau_5.xip,
                tau_5.varxip,
                tau_5.xim,
                tau_5.varxim,
            ],
            names=(
                'theta',
                'tau_0_p',
                'vartau_0_p',
                'tau_0_m',
                'vartau_0_m',
                'tau_2_p',
                'vartau_2_p',
                'tau_2_m',
                'vartau_2_m',
                'tau_5_p',
                'vartau_5_p',
                'tau_5_m',
                'vartau_5_m',
            )
        )

        if self.verbose:
            print("Done...")

        if save_cov:
            if self.verbose:
                print("Computing the covariance...")

            cov = treecorr.estimate_multi_cov([tau_0, tau_2, tau_5], var_method, func)

            np.save(self.catalogs._output+'/'+'cov_tau_'+catalog_id, cov)

        self.save_tau_stats(filename) #A bit dirty just because of consistency of the datatype :/
        self.load_tau_stats(filename)

    def save_tau_stats(self, filename):
        self.tau_stats.write(self.catalogs._output+'/'+filename, format='fits', overwrite=True)

    def load_tau_stats(self, filename):
        self.tau_stats = fits.getdata(self.catalogs._output+'/'+filename)

    def plot_tau_stats(self, filenames, colors, catalog_ids, savefig=None, plot_tau_m=True, legend="inside"):
        """
        plot_tau_stats

        Method to plot Tau + (and -) statistics of several catalogues given in argument. Figures are saved in PNG format.

        Parameters:
        ----------
        filenames : list str
            List of the files containing the rho statistics. They can be computed using the method `compute_rho_stats` of this class.

        colors : list str
            Color of the plot for the different catalogs. We recommend using different colors for different catalogs for readability.

        catalogs_id : list str
            A list of catalogs id to label accurately the legend.

        savefig : str
            If not None, saves the figure with the name given in savefig.

        plot_tau_m : bool
            If True, plot the tau - additionally.

        legend : str, optional
            allowed are "each" (default; legends in each panel), "outside" (legend outside of panels)

        Return
        ------
        fig : Figure

        ax : Axes
        """

        nrows=1 + plot_tau_m

        fig, ax = plt.subplots(nrows=nrows, ncols=3, figsize=(15,6))

        if nrows==1:
            ax = ax.reshape(1, 3)

        for filename, color, cat_id in zip(filenames, colors, catalog_ids): #Plot for the different catalogs
            self.load_tau_stats(filename)

            for i in range(3):
                for j in range(nrows):
                    p_or_m = 'm' if j else 'p'
                    p_or_m_label = '-' if j else '+'
                    xlabel=r"$\theta$ [arcmin]" if (j==nrows-1) else ''
                    if legend == "inside":
                        ylabel = r"$\tau-$statistics" if (i==0) else ''
                        label = rf'$\tau_{{{int(0.5*i**2+1.5*i)}, {p_or_m_label}}}(\theta)$ '+cat_id if i==0 else rf'$\tau_{{{int(0.5*i**2+1.5*i)}, {p_or_m_label}}}(\theta)\theta$ '+cat_id
                    else:
                        ylabel = rf"$\tau_{i}(\theta)$"
                        label = rf"$\tau_i$ {cat_id}"
                    factor_theta = np.ones_like(self.tau_stats["theta"]) if i==0 else self.tau_stats["theta"]
                    y = self.tau_stats['tau_'+str(int(0.5*i**2+1.5*i))+'_'+p_or_m]*factor_theta
                    yerr_in = np.sqrt(self.tau_stats['vartau_'+str(int(0.5*i**2+1.5*i))+'_'+p_or_m])*factor_theta

                    ax[j, i].errorbar(self.tau_stats["theta"], y, yerr=yerr_in, label=label, color=color, capsize=2)
                    ax[j, i].set_xlim(self._treecorr_config["min_sep"], self._treecorr_config["max_sep"])
                    ax[j, i].set_xlabel(xlabel)
                    ax[j, i].set_ylabel(ylabel)
                    ax[j, i].set_xscale('log')
                    if legend == "inside":
                        ax[j, i].legend(loc='best', fontsize='small')

        if legend == "outside":
            ax[-1, -1].legend(bbox_to_anchor=(1.5, 0.0), fontsize='small')

        plt.tight_layout()
        if savefig is not None:
            plt.savefig(self.catalogs._output+'/'+savefig, bbox_inches='tight')

        return fig, ax

class PSFErrorFit():
    """
    PSFErrorFit

    This class uses rho and tau statistics data to fit parameters alpha, beta and eta in the psf error model
    (Gatti et al. 2022). It includes methods to plot the best fit value and the tau statistics as well as
    as the systematic error for different catalogs.
    A Likelihood-Based Inference is used assuming a Gaussian likelihood.
    """

    def __init__(self, rho_stat_handler, tau_stat_handler, data_directory):

        self.rho_stat_handler = rho_stat_handler
        self.tau_stat_handler = tau_stat_handler
        print("Class created. Don't forget to load your rho and tau statistics and define your prior and your likelihood.")
        self.cov_rho = None
        self.cov_tau = None
        self.init_log_prior()

        def log_likelihood(theta, y, inv_cov):
            y_model = self.model(theta)
            d = y_model -y
            return -0.5 * d.T@inv_cov@d

        self.log_likelihood = log_likelihood
        self.data_directory = data_directory

        self.rho_stat_handler.catalogs._output = self.data_directory #Change the path to the specified data directory
        self.tau_stat_handler.catalogs._output = self.data_directory

    def set_data_directory(self, data_directory):
        """
        set_data_directory

        Sets the value of the data directory to `data_directory`

        Parameters
        ----------
        data_directory : str
            The new data directory.
        """
        self.data_directory = data_directory
        self.rho_stat_handler.catalogs._output = self.data_directory #Change the path to the specified data directory
        self.tau_stat_handler.catalogs._output = self.data_directory


    def load_rho_stat(self, filename):
        """
        load_rho_stats

        Load a file containing the rho stats in the RhoStat class.
        Be aware that the theta points should match those of the TauStat class.

        Parameters
        ----------
        filename : str
            Name of the file containing the rho statistics.
        """
        self.rho_stat_handler.load_rho_stats(filename)

    def load_tau_stat(self, filename):
        """
        load_tau_stat

        Load a file containing the tau stats in the TauStat class.
        Be aware that the theta points should match those of the RhoStat class.

        Parameters
        ----------
        filename : str
            Name of the file containing the rho statistics.
        """
        self.tau_stat_handler.load_tau_stats(filename)

    def load_covariance(self, filename, cov_type='rho'):
        """
        load_covariance

        Load a covariance matrix to run the inference.

        Parameters
        ----------
        filename : str
            Name of the file containing the covariance matrix.
        """
        if cov_type=='rho':
            self.cov_rho = np.load(self.data_directory+'/'+filename)
        else:
            self.cov_tau = np.load(self.data_directory+'/'+filename)

    def init_log_prior(self, low_alpha=-2.0, high_alpha=2.0, low_beta=-10.0, high_beta=10.0, low_eta=-20.0, high_eta=20.0):
        """
        init_log_prior

        Initialise the prior used in the inference. A flat prior is used for all parameters.
        The bounds can be specified.

        Parameters
        ----------
        low_alpha : float
            Lower bound on alpha prior (Default: -2.0).

        high_alpha : float
            Upper bound on alpha prior (Default: 2.0).

        low_beta : float
            Lower bound on beta prior (Default: -10.0).

        high_beta : float
            Upper bound on beta prior (Default: 10.0).

        low_eta : float
            Lower bound on eta prior (Default: -20.0).

        high_eta : float
            Upper bound on eta prior (Default: 20.0).

        Returns
        -------
        function
            Store in the attribute `log_prior` of the class a function that returns the value of the log_prior
            given a set of parameters theta.
        """
        def log_prior(theta):
            alpha, beta, eta = theta
            if low_alpha <= alpha <= high_alpha and low_beta <= beta <= high_beta and low_eta <=eta <= high_eta:
                return 0.0
            return -np.inf

        self.log_prior = log_prior

    def model(self, theta):
        """
        model

        Implements the systematic error. It outputs the tau+ statistics given the rho statistics and a set of parameters theta.

        Parameters
        ----------

        theta : tuple
            Parameters (alpha, beta, eta) of the model.

        Returns
        -------
        np.array
            A flattened array containing the tau+ statistics.
        """
        alpha, beta, eta = theta

        rhos = self.rho_stat_handler.rho_stats
        tau_0_p = alpha * rhos["rho_0_p"] + beta * rhos["rho_2_p"] + eta * rhos["rho_5_p"]
        tau_2_p = alpha * rhos["rho_2_p"] + beta * rhos["rho_1_p"] + eta * rhos["rho_4_p"]
        tau_5_p = alpha * rhos["rho_5_p"] + beta * rhos["rho_4_p"] + eta * rhos["rho_3_p"]

        model_output = np.array([
            tau_0_p,
            tau_2_p,
            tau_5_p
        ])

        return model_output.flatten()

    def log_probability(self, theta, y, inv_cov):
        """
        log_probability

        Computes the log probability given a set of parameters, input, output and the inverse of the covariance matrix

        Parameters
        ----------
        theta : tuple
            Parameters (alpha, beta, eta) of the model.

        y : np.array
            Output data containing the tau statistics to be fitted.

        inv_cov : np.array
            Inverse of the covariane matrix.
        """

        lp = self.log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.log_likelihood(theta, y, inv_cov)

    def run_chain(self, init=np.array([0.0,0.0,0.0]), nwalkers=124, nsamples=10000, discard=300, thin=100, verbose=True, savefig=None, npatch=200, apply_debias=False):
        """
        run_chain

        Run an MCMC analysis to fit the parameters alpha, beta and eta using rho and tau statistics.
        Emcee package is used to run the inference.

        Parameters
        ----------
        init : np.array
            Initial value of the parameters. An additional random noise will be added. (Default: [0,0,0])

        nwalkers : int
            Number of walkers used in the Ensemble Sampler (See emcee documentation). (Default: 124)

        nsamples : int
            Number of samples to draw (Default: 10000).

        discard : int
            Number of samples discarded in the burn-in phase (Default: 300).

        thin : int
            Number of samples thinned out (Default: 100).

        verbose : bool
            If True, prints several informations.

        npatch : int
            The number of patches used to perform the inference. (Only used if apply_bias if true, Default: 200)

        apply_debias : bool
            If True, apply some debiasing of the inverse of the covariance matrix. (Default: False)

        Returns
        -------

        np.array:
            Array of samples obtained from the MCMC analysis.

        np.array:
            Best-fit values of the parameters

        np.array:
            Error bars at the 68% confidence level.
        """
        ndim = 3
        assert (self.rho_stat_handler.rho_stats is not None), ("Please load rho statistics data.") #Check if data was loaded
        assert (self.tau_stat_handler.tau_stats is not None), ("Please load tau statistics data.")
        #assert (np.all(self.rho_stat_handler.rho_stats["theta"] == self.tau_stat_handler.tau_stats["theta"])), ("The rho and tau statistics have not the same angular scales. Check that they come from the same catalog with the same treecorr config.")
        #Check that the abssiss are the same

        assert (self.cov_tau is not None), ("Please load a covariance matrix")

        inv_cov = np.linalg.inv(self.cov_tau)
        output = np.array([
            self.tau_stat_handler.tau_stats["tau_0_p"],
            self.tau_stat_handler.tau_stats["tau_2_p"],
            self.tau_stat_handler.tau_stats["tau_5_p"]
        ]).flatten()

        if apply_debias:
            inv_cov = (npatch - output.shape[0] - 2)/(npatch-1)*inv_cov

        init = init + 1e-1*np.random.randn(nwalkers, ndim)

        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, self.log_probability, args=(output, inv_cov)
        )

        print("Run MCMC analysis...")
        sampler.run_mcmc(init, nsamples, progress=verbose)
        print("Done")

        tau = sampler.get_autocorr_time()

        if verbose:
            print("Autocorrelation-time:")
            print(tau)

        labels = [r"$\alpha$", r"$\beta$", r"$\eta$"]

        if savefig is not None:
            fig, axes = plt.subplots(3, figsize=(10, 7), sharex=True) #Result completely unconstrained. have another look at the covariance matrix
            samples = sampler.get_chain()
            for i in range(ndim):
                ax = axes[i]
                ax.plot(samples[:, :, i], "k", alpha=0.3)
                ax.set_xlim(0, len(samples))
                ax.set_ylabel(labels[i])
                ax.yaxis.set_label_coords(-0.1, 0.5)

                axes[-1].set_xlabel("step number");

            plt.savefig(self.data_directory+'/'+savefig)

        flat_samples = sampler.get_chain(discard=discard, thin=thin, flat=True)
        mcmc_result, q = self.get_mcmc_from_samples(flat_samples)

        #mcmc_result = np.percentile(flat_samples, [16, 50, 84], axis=0)
        #q = np.diff(mcmc_result, axis=0)
        if verbose:
            print(f"Number of samples: {flat_samples.shape[0]}\n")

            print("Parameters constraints")
            print("----------------------")
            for i in range(ndim):
                print('Parameter: '+labels[i]+f'={mcmc_result[1, i]:.4f}^+{q[0, i]:.4f}_{q[1, i]:.4f}')

            print(f"Max log_likelihood: {self.log_likelihood(mcmc_result[1,:], output, inv_cov)}")

        return flat_samples, mcmc_result, q

    def get_sample_path(self, catalog_id):
        """Get Sample Path.

        Return file path of MCMC sample.

        Parameters
        ----------
        catalog_id : str
            ID of the catalog used for this run

        Returns
        -------
        str
            file path
        """
        file_path = f"{self.rho_stat_handler.catalogs._output}/samples_{catalog_id}.npy"
        return file_path
    
    def get_params_path(self, catalog_id):
        """Get Params Path.

        Return file path of parameters.

        Parameters
        ----------
        catalog_id : str
            ID of the catalog used for this run

        Returns
        -------
        str
            file path
        """
        file_path = f"{self.rho_stat_handler.catalogs._output}/params_{catalog_id}.npy"
        return file_path


    def save_samples(self, flat_samples, catalog_id):
        """Save Samples.

        Save samples to file.

        Parameters
        ----------
        flat_samples : np.array
            Samples obtained from the MCMC analysis
        catalog_id : str
            ID of the catalog used for this run

        """
        np.save(self.get_sample_path(catalog_id), flat_samples)

    def load_samples(self, catalog_id):
        """Load Samples.

        Load samples from file.

        Parameters
        ----------
        Returns
        -------
        np.array
            Samples obtained from the MCMC analysi

        """
        file_path = self.get_sample_path(catalog_id)
        flat_samples = np.load(file_path)

        return flat_samples


    def get_mcmc_from_samples(self, flat_samples):
        """Get MCMC From Samples.

        Return MCMC samples and quanties from samples.

        Parameters
        ----------
        flat_samples : np.array
            Samples obtained from the MCMC analysis

        Returns
        -------
        np.array
            Best-fit values of the parameters

        np.array
            Error bars at the 68% confidence level.

        """
        mcmc_result = np.percentile(flat_samples, [16, 50, 84], axis=0)
        q = np.diff(mcmc_result, axis=0)

        return mcmc_result, q
    
    def build_rho_matrix(self, rho=None):
        """
        Build a matrix of rho statistics to get the tau statistics when multiplying with the parameters.

        Parameters
        ----------
        rho : np.array
            Rho statistics data. If None, use the data stored in the RhoStat class.

        Returns
        -------
        np.array
            Matrix of rho statistics.
        """
        n_thetas = len(self.rho_stat_handler.rho_stats["theta"]) #number of bins
        rho_matrix = np.zeros((3*n_thetas, 3))
        if rho is None:
            rho_stats = self.rho_stat_handler.rho_stats
            for i in range(n_thetas):
                rho_matrix[i] = [rho_stats["rho_0_p"][i], rho_stats["rho_2_p"][i], rho_stats["rho_5_p"][i]]
                rho_matrix[i+n_thetas] = [rho_stats['rho_2_p'][i], rho_stats['rho_1_p'][i], rho_stats['rho_4_p'][i]]
                rho_matrix[i+2*n_thetas] = [rho_stats['rho_5_p'][i], rho_stats['rho_4_p'][i], rho_stats['rho_3_p'][i]]
        else:
            rho_stats = rho
            for i in range(n_thetas):
                rho_matrix[i] = [rho_stats[0, i], rho_stats[2, i], rho_stats[5, i]]
                rho_matrix[i+n_thetas] = [rho_stats[2, i], rho_stats[1, i], rho_stats[4, i]]
                rho_matrix[i+2*n_thetas] = [rho_stats[5, i], rho_stats[4, i], rho_stats[3, i]]
        return rho_matrix
    
    def build_tau_vec(self, tau=None):
        """
        Build a vector of tau statistics to get the least squares parameters.

        Parameters
        ----------
        tau : np.array
            Tau statistics data. If None, use the data stored in the TauStat class.
        
        Returns
        -------
        np.array
            Vector of tau statistics.
        """
        if tau is None:
            tau_stats = self.tau_stat_handler.tau_stats
            tau_vec = np.array([tau_stats["tau_0_p"], tau_stats["tau_2_p"], tau_stats["tau_5_p"]]).flatten()
        else:
            tau_vec = tau.flatten()
        return tau_vec
    
    def get_least_squares_params(self, npatch=200, apply_debias=False):
        """
        Compute the least square optimum of the residuals for the mean value measured by TreeCorr.

        Parameters
        ----------
        npatch : int
            The number of patches used to compute the covariance. (Default: 200)
        apply_debias : bool
            If True, apply some debiasing of the inverse of the covariance matrix. (Default: False)

        Returns
        -------
        np.array
            Parameters realising the optimum.
        """
        rho_matrix = self.build_rho_matrix()
        tau_vec = self.build_tau_vec()
        assert self.cov_tau is not None, "Please load a covariance matrix for the tau statistics."
        inv_cov = np.linalg.inv(self.cov_tau)
        if apply_debias:
            inv_cov = (npatch - tau_vec.shape[0] - 2)/(npatch-1)*inv_cov
        return np.linalg.inv(rho_matrix.T @ inv_cov @ rho_matrix) @ rho_matrix.T @ inv_cov @ tau_vec
    
    def get_least_squares_params_samples(self, npatch, apply_debias=False, n_samples=10000, verbose=True):
        """
        Computes the least square optimum of the residuals by sampling rho and tau statistics from their covariance.

        Parameters
        ----------
        npatch : int
            The number of patches used to compute the covariance.
        apply_debias : bool
            If True, apply some debiasing of the inverse of the covariance matrix. (Default: False)
        n_samples : int
            The number of samples to draw from the covariance matrix.
        verbose : bool
            If True, prints several informations.

        Returns
        -------
        np.array
            Samples obtained from the MCMC analysis
        np.array
            Best-fit values of the parameters
        np.array
            Error bars at the 68% confidence level.
        """
        assert self.cov_tau is not None, "Please load a covariance matrix for the tau statistics."
        assert self.cov_rho is not None, "Please load a covariance matrix for the rho statistics."
        rho_stats = self.rho_stat_handler.rho_stats
        rho_mean = np.array([rho_stats["rho_0_p"], rho_stats["rho_1_p"], rho_stats["rho_2_p"], rho_stats["rho_3_p"],
                             rho_stats["rho_4_p"], rho_stats["rho_5_p"]]).flatten()
        tau_stats = self.tau_stat_handler.tau_stats
        tau_mean = np.array([tau_stats["tau_0_p"], tau_stats["tau_2_p"], tau_stats["tau_5_p"]]).flatten()
        for i in tqdm(range(n_samples)):
            rho = np.random.multivariate_normal(rho_mean, self.cov_rho)
            rho = rho.reshape((6, -1))
            rho_matrix = self.build_rho_matrix(rho)
            tau = np.random.multivariate_normal(tau_mean, self.cov_tau)
            tau_vec = self.build_tau_vec(tau)
            inv_cov = np.linalg.inv(self.cov_tau)
            if apply_debias:
                inv_cov = (npatch - tau_vec.shape[0] - 2)/(npatch-1)*inv_cov
            if i==0:
                samples = np.linalg.inv(rho_matrix.T @ inv_cov @ rho_matrix) @ rho_matrix.T @ inv_cov @ tau_vec
            else:
                samples = np.vstack((samples, np.linalg.inv(rho_matrix.T @ inv_cov @ rho_matrix) @ rho_matrix.T @ inv_cov @ tau_vec))
        result, q = self.get_mcmc_from_samples(samples)

        if verbose:
            labels = [r"$\alpha$", r"$\beta$", r"$\eta$"]
            print(f"Number of samples: {samples.shape[0]}\n")

            print("Parameters constraints")
            print("----------------------")
            for i in range(3):
                print('Parameter: '+labels[i]+f'={result[1, i]:.4f}^+{q[0, i]:.4f}_{q[1, i]:.4f}')

            print(f"Chi square: {self.eval_chi_square(result[1,:], npatch=npatch, apply_debias=apply_debias)}")
        
        return samples, result, q
    
    def eval_chi_square(self, theta, npatch=200, apply_debias=False):
        """
        Compute the chi square of the fit.

        Parameters
        ----------
        theta : tuple
            Parameters (alpha, beta, eta) of the model.
        npatch : int
            The number of patches used to compute the covariance.
        apply_debias : bool
            If True, apply some debiasing of the inverse of the covariance matrix. (Default: False)

        Returns
        -------
        float
            The value of the chi square.
        """
        rho_matrix = self.build_rho_matrix()
        tau_vec = self.build_tau_vec()
        assert self.cov_tau is not None, "Please load a covariance matrix for the tau statistics."
        inv_cov = np.linalg.inv(self.cov_tau)
        if apply_debias:
            inv_cov = (npatch - tau_vec.shape[0] - 2)/(npatch-1)*inv_cov
        return (tau_vec - rho_matrix @ theta) @ inv_cov @ (tau_vec - rho_matrix @ theta)

    def save_params(self, theta, catalog_id):
        """Save Parameters.

        Save parameters to file.

        Parameters
        ----------
        theta : np.array
            Parameters obtained from the MCMC analysis or the LSQ fit.
        catalog_id : str
            ID of the catalog used for this run

        """
        np.save(self.get_sample_path(catalog_id), theta)
    
    def plot_tau_stats_w_model(self, theta, filename, color, catalog_id, savefig=None):
        """
        plot_tau_stats_w_model

        Plot the tau statistics with the value in theta as parameters.

        Parameters
        ----------
        theta : tuple
            Parameters (alpha, beta, eta).
        filename : str
            Name of the file containing the tau statistics. Take care to load the corresponding rho statistics before.
        color : str
            Color of the tau statistics lines.
        catalog_id : str
            A catalog id for the legend
        savefig : str
            If not None, save the figure with the given filename.
        """

        fig, ax = self.tau_stat_handler.plot_tau_stats([filename], [color], [catalog_id], plot_tau_m=False)

        assert (self.rho_stat_handler.rho_stats is not None), ("Please load rho statistics data.") #Check if data was loaded

        scales_diff_ratio = np.abs(
            (self.rho_stat_handler.rho_stats["theta"] -
             self.tau_stat_handler.tau_stats["theta"]
            ) / self.tau_stat_handler.tau_stats["theta"]
        )
        if np.any(scales_diff_ratio > 0.001):
            print("theta for rho: ", self.rho_stat_handler.rho_stats["theta"])
            print("theta for tau: ", self.tau_stat_handler.tau_stats["theta"])
            print(scales_diff_ratio, max(scales_diff_ratio))
            raise ValueError(
                "The rho and tau statistics have not the same angular scales: "
                + " Check that they come from the same catalog with the same"
                + " treecorr config."
            )

        taus = self.model(theta).reshape(3, -1)

        ylim = [[-4e-5,  4e-5], [-6e-6, 4e-6], [-7e-6, 1e-6]]
        for i in range(3):
            factor = np.ones_like(self.tau_stat_handler.tau_stats["theta"]) if i==0 else self.tau_stat_handler.tau_stats["theta"]
            ax[0, i].plot(self.tau_stat_handler.tau_stats["theta"], taus[i]*factor, color='red', label='Model')
            ax[0, i].legend(loc='upper right', fontsize='small')
            ax[0, i].set_ylim(ylim[i])
            ax[0, i].axhline(color="k", linestyle="dotted", linewidth=0.5)

        if savefig is not None:
            plt.savefig(self.data_directory+'/'+savefig)

        plt.close()

    def plot_xi_psf_sys(self, theta, cat_id, color, savefig=None, alpha=1):
        """
        Plot Xi Psf sys.

        Plot the systematic error on the corss-correlation given parameters (alpha, beta, eta).

        Parameters
        ----------
        theta : tuple
            Parameters (alpha, beta, eta) used to compute the systematic error.

        """
        xi_psf_sys = self.compute_xi_psf_sys(theta)
        if alpha < 1:
            plt.errorbar(self.rho_stat_handler.rho_stats["theta"], xi_psf_sys, color=color, capsize=2, alpha=alpha)
        else:
            plt.errorbar(self.rho_stat_handler.rho_stats["theta"], xi_psf_sys, color=color, capsize=2, alpha=alpha, label=r'$\xi^{\rm PSF}_{\rm sys, +}$ '+cat_id)

        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel(r"$\theta$ [arcmin]")
        plt.ylabel(r"$\xi^{\rm PSF}_{\rm sys}$")

        if savefig:
            plt.savefig('xi_psf_sys.png')
            plt.close()


    def plot_xi_psf_sys_terms(self, cat_id, theta, out_path, yscale="log"):

        ls = ["dotted", "dashed", "dashdot", (-1, (3, 5, 1, 5, 1, 5)), (0, (1, 10)), (0, (5, 5))]
        color = ["green", "blue", "red", "magenta", "cyan", "orange"]

        plt.figure(figsize=(15, 6))

        #self.plot_xi_psf_sys(theta, cat_id, "black")
        ang_scales = self.rho_stat_handler.rho_stats["theta"]

        label_pre = [
            r"$\alpha^2$",
            r"$\beta^2$",
            r"$\eta^2$",
            r"$2 \alpha \beta$",
            r"$2 \alpha \eta$",
            r"$2 \beta \eta$",
        ]
        xi_sum = np.zeros_like(ang_scales)

        if yscale == "linear":
            plot_fct = plt.semilogx
            ylim = [-0.5e-6, 5e-6]
        else:
            plot_fct = plt.loglog
            ylim = [3e-10, 5e-6]

        linewidth_pos = 3
        linewidth_neg = 1

        xi_psf_sys = self.compute_xi_psf_sys(theta)
        plot_fct(
            ang_scales,
            xi_psf_sys,
            linestyle="-",
            linewidth=linewidth_pos,
            color="black",
            label=r'$\xi^{\rm PSF}_{\rm sys, +}$ '+ cat_id,
        )

        for term in range(6):
            xi_psf_sys_term = self.compute_xi_psf_sys_term(theta, term)
            xi_sum += xi_psf_sys_term
            if xi_psf_sys_term[-1] > 0:
                linewidth = linewidth_pos
            else:
                linewidth = linewidth_neg
            plot_fct(
                ang_scales,
                np.abs(xi_psf_sys_term),
                linestyle=ls[term],
                linewidth=linewidth,
                color=color[term],
                label=fr"{label_pre[term]} $\rho_{term}$",
            )
        xi_psf_sys_check = self.compute_xi_psf_sys(theta)

        plot_fct(ang_scales, xi_psf_sys_check, "bo", mfc="none")
        plot_fct(ang_scales, xi_sum, "bs", mfc="none")

        plt.xlabel(r"$\theta$ [arcmin]")
        plt.ylabel(r"$\xi^{\rm PSF}_{\rm sys}$")
        plt.legend(loc="best", fontsize="small")
        plt.ylim(ylim)
        plt.tight_layout()
        plt.savefig(out_path, bbox_inches='tight')
        plt.close()


    def compute_xi_psf_sys_term(self, theta, term):
        """
        Compute Xi Psf Sys Term.

        Compute one term of the systematic error on the cross-correlation given parameters (alpha, beta, eta)

        Parameters
        ----------
        theta : tuple
            Parameters (alpha, beta, eta) used to compute the systematic error

        term : int
            term (rho function) number, from 0 to 5

        Return
        ------
        np.array
            xi_psf_sys

        """
        alpha, beta, eta = theta

        if term == 0:
            prefactor = alpha ** 2
        elif term == 1:
            prefactor = beta ** 2
        elif term == 3:
            prefactor = eta ** 2
        elif term == 2:
            prefactor = 2 * alpha * beta
        elif term == 5:
            prefactor = 2 * alpha * eta
        elif term == 4:
            prefactor = 2 * beta * eta
        else:
            raise ValueError(f"Invalid term {term}")

        return prefactor * self.rho_stat_handler.rho_stats[f"rho_{term}_p"]

    def compute_xi_psf_sys(self, theta):
        """
        Compute Xi Psf Sys.

        Compute the systematic error on the cross-correlation given parameters (alpha, beta, eta)

        Parameters
        ----------
        theta : tuple
            Parameters (alpha, beta, eta) used to compute the systematic error

        Return
        ------
        np.array
            xi_psf_sys
        """

        alpha, beta, eta = theta

        xi_psf_sys = np.zeros_like(self.rho_stat_handler.rho_stats["theta"])

        for term in range(6):
           xi_psf_sys += self.compute_xi_psf_sys_term(theta, term)

            #alpha ** 2 * self.rho_stat_handler.rho_stats["rho_0_p"]
            #+ beta ** 2 * self.rho_stat_handler.rho_stats["rho_1_p"]
            #+ eta ** 2 * self.rho_stat_handler.rho_stats["rho_3_p"]
            #+ 2 * alpha * beta * self.rho_stat_handler.rho_stats["rho_2_p"]
            #+ 2 * alpha * eta * self.rho_stat_handler.rho_stats["rho_5_p"]
            #+ 2 * beta * eta * self.rho_stat_handler.rho_stats["rho_4_p"]

        return xi_psf_sys
