"""
This module sets up a class to compute the rho stats computation

Author: Sacha Guerrini
"""

import warnings

import emcee
import matplotlib.pyplot as plt
import numpy as np
import treecorr
from astropy.io import fits
from astropy.table import Table
from tqdm import tqdm


def neg_dash(
    ax,
    x_in,
    y_in,
    yerr_in,
    vertical_lines=True,
    xlabel="",
    ylabel="",
    rho_nb="",
    tau_nb="",
    cat_id="",
    ylim=None,
    semilogx=False,
    semilogy=False,
    **kwargs,
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
    if "linestyle" in kwargs.keys():
        print(
            "Warning: linestyle was provided but that would kind of defeat"
            + "the purpose, so I will just ignore it. Sorry."
        )
        del safekwargs["linestyle"]
    for errorbar_kword in [
        "fmt",
        "ecolor",
        "elinewidth",
        "capsize",
        "barsabove",
        "errorevery",
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
                linestyle="-",
                **errbkwargs,
            )
            if vertical_lines:
                ax.vlines(
                    x[first_change - 1],
                    0,
                    y[first_change - 1],
                    linestyle="-",
                    **safekwargs,
                )
                ax.vlines(
                    x[first_change],
                    0,
                    np.abs(y[first_change]),
                    linestyle="--",
                    **safekwargs,
                )
        else:
            ax.errorbar(
                x[:first_change],
                np.abs(y[:first_change]),
                yerr=yerr[:first_change],
                linestyle="--",
                **errbkwargs,
            )
            if vertical_lines:
                ax.vlines(
                    x[first_change - 1],
                    0,
                    np.abs(y[first_change - 1]),
                    linestyle="--",
                    **safekwargs,
                )
                ax.vlines(
                    x[first_change],
                    0,
                    y[first_change],
                    linestyle="-",
                    **safekwargs,
                )
        x = x[first_change:]
        y = y[first_change:]
        yerr = yerr[first_change:]
        current_sign *= -1
        first_change = np.argmax(current_sign * y < 0)
    # one last time when `first_change'==0 ie no more changes:
    if rho_nb:
        lab = rf"$\rho_{rho_nb}(\theta)$ " + cat_id
    elif tau_nb:
        lab = rf"$\tau_{tau_nb}(\theta)$" + cat_id
    else:
        lab = cat_id
    if current_sign > 0:
        ax.errorbar(x, y, yerr=yerr, linestyle="-", label=lab, **errbkwargs)
    else:
        ax.errorbar(x, np.abs(y), yerr=yerr, linestyle="--", label=lab, **errbkwargs)
    if semilogx:
        ax.set_xscale("log")
    if semilogy:
        ax.set_yscale("log")
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)


class Catalogs:
    """
    Catalogs

    Class to build the different treecorr catalogs given a shape catalog that will be
    used to compute the different statistics
    """

    def __init__(self, params=None, output=None):
        # set default parameters
        if params is None:
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
            "ra_PSF_col": "RA",
            "dec_PSF_col": "Dec",
            "e1_PSF_col": "HSM_G1_PSF",
            "e2_PSF_col": "HSM_G2_PSF",
            "e1_star_col": "HSM_G1_STAR",
            "e2_star_col": "HSM_G2_STAR",
            "PSF_size": "HSM_T_PSF",
            "star_size": "HSM_T_STAR",
            "M_4_1_psf_col": "M_4_PSF_1",
            "M_4_2_psf_col": "M_4_PSF_2",
            "M_4_1_star_col": "M_4_STAR_1",
            "M_4_2_star_col": "M_4_STAR_2",
            "PSF_flag": "HSM_FLAG_PSF",
            "star_flag": "HSM_FLAG_STAR",
            "patch_number": 120,
            "ra_units": "deg",
            "dec_units": "deg",
        }

        if output is not None:
            self._output = output
        else:
            self._output = "."

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
        assert (path_gal is not None) or (path_psf is not None), (
            "Please specify a path for the shear catalog you want to read."
        )
        if path_gal is not None:
            dat_shear = fits.getdata(path_gal, ext=hdu)
            return dat_shear
        if path_psf is not None:
            dat_psf = fits.getdata(path_psf, ext=hdu)
            return dat_psf

    def get_cat_fields(self, cat, cat_type):
        """
        Get Cat Fields

        Get catalogue fields for correlation.

        Parameters
        ----------
        cat : str
            catalogue of galaxies or stars. Type should match the cat_type given in argument.
        cat_type : str
            catalogue type, allowed are 'gal', 'psf', 'psf_error' or 'psf_size_error'

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

        allowed_types = [
            "gal",
            "psf",
            "psf_error",
            "psf_size_error",
            "psf_fourth_moment",
            "psf_fourth_moment_error",
        ]

        assert cat_type in allowed_types, (
            "The specified catalogue type is invalid. Check the one you use is allowed."
            "Allowed cat_type: 'gal', 'psf', 'psf_error', 'psf_size_error', 'psf_fourth_moment', 'psf_fourth_moment_error'."
        )

        if cat_type == "gal":
            ra = cat[self._params["ra_col"]]
            dec = cat[self._params["dec_col"]]
            if self._params["w_col"] is not None:
                weights = cat[self._params["w_col"]]
            else:
                weights = np.ones_like(ra)
            g1 = cat[self._params["e1_col"]] - np.average(
                cat[self._params["e1_col"]], weights=weights
            )
            g2 = cat[self._params["e2_col"]] - np.average(
                cat[self._params["e2_col"]], weights=weights
            )
            if self._params.get("R11", None) is not None:
                g1 /= self._params["R11"]
            if self._params.get("R22", None) is not None:
                g2 /= self._params["R22"]
        else:
            # Add a mask?
            # mask = (self.dat_psf[self._params["HSM_FLAG_PSF"]]==0) & (self.dat_psf[self._params["HSM_FLAG_STAR"]]==0)
            if (
                self._params["ra_PSF_col"] is not None
            ):  # Check if a name for the columns of the PSF coordinates is given
                ra = cat[self._params["ra_PSF_col"]]
                dec = cat[self._params["dec_PSF_col"]]
            else:  # Else takes the same than the galaxy catalogue
                ra = cat[self._params["ra_col"]]
                dec = cat[self._params["dec_col"]]
            weights = None

            if cat_type == "psf":
                g1 = cat[
                    self._params["e1_PSF_col"]
                ]  # - cat[self._params["e1_PSF_col"]].mean()
                g2 = cat[
                    self._params["e2_PSF_col"]
                ]  # - cat[self._params["e2_PSF_col"]].mean()

            elif cat_type == "psf_error":
                g1 = cat[self._params["e1_star_col"]] - cat[self._params["e1_PSF_col"]]
                # g1 -= g1.mean()
                g2 = cat[self._params["e2_star_col"]] - cat[self._params["e2_PSF_col"]]
                # g2 -= g2.mean()

            elif cat_type == "psf_size_error":
                size_star = cat[self._params["star_size"]]
                size_psf = cat[self._params["PSF_size"]]

                g1 = (
                    cat[self._params["e1_star_col"]]
                    * (size_star - size_psf)
                    / size_star
                )
                # g1 -= g1.mean()
                g2 = (
                    cat[self._params["e2_star_col"]]
                    * (size_star - size_psf)
                    / size_star
                )
                # g2 -= g2.mean()

            elif cat_type == "psf_fourth_moment":
                g1 = cat[self._params["M_4_1_psf_col"]]
                g2 = cat[self._params["M_4_2_psf_col"]]

            elif cat_type == "psf_fourth_moment_error":
                g1 = (
                    cat[self._params["M_4_1_star_col"]]
                    - cat[self._params["M_4_1_psf_col"]]
                )
                g2 = (
                    cat[self._params["M_4_2_star_col"]]
                    - cat[self._params["M_4_2_psf_col"]]
                )

        return ra, dec, g1, g2, weights

    def build_catalog(
        self,
        cat,
        cat_type,
        key,
        npatch=None,
        patch_centers=None,
        mask=None,
    ):
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

        mask : np.array
            A mask array to select only the relevant objects in the catalogue. If None, no mask is applied. (Default: None)
        """

        if npatch is None:
            npatch = self._params["patch_number"]

        ra, dec, g1, g2, weights = self.get_cat_fields(cat, cat_type)

        if mask is None:
            mask = np.ones_like(ra, dtype=bool)

        if patch_centers is None:
            cat_tc = treecorr.Catalog(
                ra=ra[mask],
                dec=dec[mask],
                g1=g1[mask],
                g2=g2[mask],
                w=weights[mask] if weights is not None else None,
                ra_units=self._params["ra_units"],
                dec_units=self._params["dec_units"],
                npatch=npatch,
            )
        else:
            cat_tc = treecorr.Catalog(
                ra=ra[mask],
                dec=dec[mask],
                g1=g1[mask],
                g2=g2[mask],
                w=weights[mask] if weights is not None else None,
                ra_units=self._params["ra_units"],
                dec_units=self._params["dec_units"],
                patch_centers=patch_centers,
            )

        self.catalogs_dict.update({key: cat_tc})

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


class RhoStat:
    """
    RhoStat

    Class to compute the rho statistics (Rowe 2010) of a PSF catalogue.

    Parameters
    ----------
    params : dict
        Dictionary containing the parameters of the class. If None, default parameters are used.
    output : str
        Path to the output directory. Default is the current directory.
    use_eta : bool
        If True, add correlations with size residuals of the PSF. Default is True.
    scalar_eta : bool
        If True, compute correlations with size residuals as correlations between a spin-2 field and a
        scalar field (GKCorrelator in treecorr). Default is False.
    treecorr_config : dict
        Dictionary containing the configuration of the treecorr.GGCorrelation. Default is None.
    verbose : bool
        If True, print messages to the terminal. Default is False.
    """

    def __init__(
        self,
        params=None,
        output=None,
        use_eta=True,
        scalar_eta=False,
        use_fourth_moment=False,
        treecorr_config=None,
        verbose=False,
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
                "var_method": "jackknife",
            }
        else:
            self._treecorr_config = treecorr_config
        self.use_eta = use_eta
        self.scalar_eta = scalar_eta
        self.use_fourth_moment = use_fourth_moment

        if self.scalar_eta and not self.use_eta:
            print(
                "Warning: scalar_eta is set to True but use_eta is set to False. Setting use_eta to True."
            )
            self.use_eta = True

        self.verbose = verbose

    def build_cat_to_compute_rho(self, path_cat_star, catalog_id="", mask=None, hdu=1):
        """
        build_cat_to_compute_rho

        Parameters
        ----------
        path_cat_star : str
            Path to the catalog of stars used to compute the rho-statistics.

        catalog_id : str
            An id to identify the catalog used in the keys of the stored treecorr.Catalog.

        mask : np.array
            A mask array to select only the relevant objects in the catalogue. If None, no mask is applied. (Default: None)

        hdu : int, optional
            HDU number of input FITS file, default is 1
        """

        psf_cat = self.catalogs.read_shear_cat(
            path_gal=None, path_psf=path_cat_star, hdu=hdu
        )

        if self.verbose:
            print("Building catalogs...")

        self.catalogs.build_catalog(
            cat=psf_cat,
            cat_type="psf",
            key="psf_" + catalog_id,
            mask=mask,
        )
        patch_centers = self.catalogs.catalogs_dict["psf_" + catalog_id].patch_centers
        self.catalogs.build_catalog(
            cat=psf_cat,
            cat_type="psf_error",
            key="psf_error_" + catalog_id,
            patch_centers=patch_centers,
            mask=mask,
        )
        if self.use_eta:
            self.catalogs.build_catalog(
                cat=psf_cat,
                cat_type="psf_size_error",
                key="psf_size_error_" + catalog_id,
                patch_centers=patch_centers,
                mask=mask,
            )

        if self.use_fourth_moment:
            self.catalogs.build_catalog(
                cat=psf_cat,
                cat_type="psf_fourth_moment",
                key="psf_fourth_moment_" + catalog_id,
                patch_centers=patch_centers,
                mask=mask,
            )
            self.catalogs.build_catalog(
                cat=psf_cat,
                cat_type="psf_fourth_moment_error",
                key="psf_fourth_moment_error_" + catalog_id,
                patch_centers=patch_centers,
                mask=mask,
            )

        del psf_cat

        if self.verbose:
            print("Catalogs successfully built...")
            self.catalogs.show_catalogs()

    def compute_rho_stats(
        self,
        catalog_id,
        filename,
        save_cov=False,
        func=None,
        var_method="jackknife",
        write_treecorr=False,
        write_patch_results=False,
        write_cov=False,
    ):
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
            print(
                "Computation of the rho statistics of " + catalog_id + " in progress..."
            )
        if write_treecorr:
            # Remove extension of the filename
            filename_no_ext = filename.split(".")[0]

        rho_0 = treecorr.GGCorrelation(self._treecorr_config)
        rho_0.process(
            self.catalogs.get_cat("psf_" + catalog_id),
            self.catalogs.get_cat("psf_" + catalog_id),
        )
        rho_1 = treecorr.GGCorrelation(self._treecorr_config)
        rho_1.process(
            self.catalogs.get_cat("psf_error_" + catalog_id),
            self.catalogs.get_cat("psf_error_" + catalog_id),
        )
        rho_2 = treecorr.GGCorrelation(self._treecorr_config)
        rho_2.process(
            self.catalogs.get_cat("psf_error_" + catalog_id),
            self.catalogs.get_cat("psf_" + catalog_id),
        )
        if write_treecorr:
            self.save_treecorr(
                rho_0, filename_no_ext + "_rho_0.txt", write_patch_results, write_cov
            )
            self.save_treecorr(
                rho_1, filename_no_ext + "_rho_1.txt", write_patch_results, write_cov
            )
            self.save_treecorr(
                rho_2, filename_no_ext + "_rho_2.txt", write_patch_results, write_cov
            )

        if self.use_eta:
            rho_3 = treecorr.GGCorrelation(self._treecorr_config)
            rho_3.process(
                self.catalogs.get_cat("psf_size_error_" + catalog_id),
                self.catalogs.get_cat("psf_size_error_" + catalog_id),
            )
            rho_4 = treecorr.GGCorrelation(self._treecorr_config)
            rho_4.process(
                self.catalogs.get_cat("psf_error_" + catalog_id),
                self.catalogs.get_cat("psf_size_error_" + catalog_id),
            )
            rho_5 = treecorr.GGCorrelation(self._treecorr_config)
            rho_5.process(
                self.catalogs.get_cat("psf_" + catalog_id),
                self.catalogs.get_cat("psf_size_error_" + catalog_id),
            )
            if write_treecorr:
                self.save_treecorr(
                    rho_3,
                    filename_no_ext + "_rho_3.txt",
                    write_patch_results,
                    write_cov,
                )
                self.save_treecorr(
                    rho_4,
                    filename_no_ext + "_rho_4.txt",
                    write_patch_results,
                    write_cov,
                )
                self.save_treecorr(
                    rho_5,
                    filename_no_ext + "_rho_5.txt",
                    write_patch_results,
                    write_cov,
                )

        if self.use_fourth_moment:
            rho_6 = treecorr.GGCorrelation(self._treecorr_config)
            rho_6.process(
                self.catalogs.get_cat("psf_fourth_moment_" + catalog_id),
                self.catalogs.get_cat("psf_fourth_moment_" + catalog_id),
            )
            rho_7 = treecorr.GGCorrelation(self._treecorr_config)
            rho_7.process(
                self.catalogs.get_cat("psf_fourth_moment_" + catalog_id),
                self.catalogs.get_cat("psf_" + catalog_id),
            )
            rho_8 = treecorr.GGCorrelation(self._treecorr_config)
            rho_8.process(
                self.catalogs.get_cat("psf_fourth_moment_" + catalog_id),
                self.catalogs.get_cat("psf_error_" + catalog_id),
            )
            rho_9 = treecorr.GGCorrelation(self._treecorr_config)
            rho_9.process(
                self.catalogs.get_cat("psf_fourth_moment_error_" + catalog_id),
                self.catalogs.get_cat("psf_fourth_moment_error_" + catalog_id),
            )
            rho_10 = treecorr.GGCorrelation(self._treecorr_config)
            rho_10.process(
                self.catalogs.get_cat("psf_fourth_moment_error_" + catalog_id),
                self.catalogs.get_cat("psf_fourth_moment_" + catalog_id),
            )
            rho_11 = treecorr.GGCorrelation(self._treecorr_config)
            rho_11.process(
                self.catalogs.get_cat("psf_fourth_moment_error_" + catalog_id),
                self.catalogs.get_cat("psf_" + catalog_id),
            )
            rho_12 = treecorr.GGCorrelation(self._treecorr_config)
            rho_12.process(
                self.catalogs.get_cat("psf_fourth_moment_error_" + catalog_id),
                self.catalogs.get_cat("psf_error_" + catalog_id),
            )

            if write_treecorr:
                self.save_treecorr(
                    rho_6,
                    filename_no_ext + "_rho_6.txt",
                    write_patch_results,
                    write_cov,
                )
                self.save_treecorr(
                    rho_7,
                    filename_no_ext + "_rho_7.txt",
                    write_patch_results,
                    write_cov,
                )
                self.save_treecorr(
                    rho_8,
                    filename_no_ext + "_rho_8.txt",
                    write_patch_results,
                    write_cov,
                )
                self.save_treecorr(
                    rho_9,
                    filename_no_ext + "_rho_9.txt",
                    write_patch_results,
                    write_cov,
                )
                self.save_treecorr(
                    rho_10,
                    filename_no_ext + "_rho_10.txt",
                    write_patch_results,
                    write_cov,
                )
                self.save_treecorr(
                    rho_11,
                    filename_no_ext + "_rho_11.txt",
                    write_patch_results,
                    write_cov,
                )
                self.save_treecorr(
                    rho_12,
                    filename_no_ext + "_rho_12.txt",
                    write_patch_results,
                    write_cov,
                )

        if self.use_eta and self.use_fourth_moment:
            rho_13 = treecorr.GGCorrelation(self._treecorr_config)
            rho_13.process(
                self.catalogs.get_cat("psf_fourth_moment_" + catalog_id),
                self.catalogs.get_cat("psf_size_error_" + catalog_id),
            )
            rho_14 = treecorr.GGCorrelation(self._treecorr_config)
            rho_14.process(
                self.catalogs.get_cat("psf_fourth_moment_error_" + catalog_id),
                self.catalogs.get_cat("psf_size_error_" + catalog_id),
            )
            if write_treecorr:
                self.save_treecorr(
                    rho_13,
                    filename_no_ext + "_rho_13.txt",
                    write_patch_results,
                    write_cov,
                )
                self.save_treecorr(
                    rho_14,
                    filename_no_ext + "_rho_14.txt",
                    write_patch_results,
                    write_cov,
                )

        if self.use_eta and not self.use_fourth_moment:
            self.rho_stats = Table(
                [
                    rho_0.meanr,
                    rho_0.xip,
                    rho_0.varxip,
                    rho_0.xim,
                    rho_0.varxim,
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
                    "theta",
                    "rho_0_p",
                    "varrho_0_p",
                    "rho_0_m",
                    "varrho_0_m",
                    "rho_1_p",
                    "varrho_1_p",
                    "rho_1_m",
                    "varrho_1_m",
                    "rho_2_p",
                    "varrho_2_p",
                    "rho_2_m",
                    "varrho_2_m",
                    "rho_3_p",
                    "varrho_3_p",
                    "rho_3_m",
                    "varrho_3_m",
                    "rho_4_p",
                    "varrho_4_p",
                    "rho_4_m",
                    "varrho_4_m",
                    "rho_5_p",
                    "varrho_5_p",
                    "rho_5_m",
                    "varrho_5_m",
                ),
            )

        elif self.use_fourth_moment and not self.use_eta:
            self.rho_stats = Table(
                [
                    rho_0.meanr,
                    rho_0.xip,
                    rho_0.varxip,
                    rho_0.xim,
                    rho_0.varxim,
                    rho_1.xip,
                    rho_1.varxip,
                    rho_1.xim,
                    rho_1.varxim,
                    rho_2.xip,
                    rho_2.varxip,
                    rho_2.xim,
                    rho_2.varxim,
                    rho_6.xip,
                    rho_6.varxip,
                    rho_6.xim,
                    rho_6.varxim,
                    rho_7.xip,
                    rho_7.varxip,
                    rho_7.xim,
                    rho_7.varxim,
                    rho_8.xip,
                    rho_8.varxip,
                    rho_8.xim,
                    rho_8.varxim,
                    rho_9.xip,
                    rho_9.varxip,
                    rho_9.xim,
                    rho_9.varxim,
                    rho_10.xip,
                    rho_10.varxip,
                    rho_10.xim,
                    rho_10.varxim,
                    rho_11.xip,
                    rho_11.varxip,
                    rho_11.xim,
                    rho_11.varxim,
                    rho_12.xip,
                    rho_12.varxip,
                    rho_12.xim,
                    rho_12.varxim,
                ],
                names=(
                    "theta",
                    "rho_0_p",
                    "varrho_0_p",
                    "rho_0_m",
                    "varrho_0_m",
                    "rho_1_p",
                    "varrho_1_p",
                    "rho_1_m",
                    "varrho_1_m",
                    "rho_2_p",
                    "varrho_2_p",
                    "rho_2_m",
                    "varrho_2_m",
                    "rho_6_p",
                    "varrho_6_p",
                    "rho_6_m",
                    "varrho_6_m",
                    "rho_7_p",
                    "varrho_7_p",
                    "rho_7_m",
                    "varrho_7_m",
                    "rho_8_p",
                    "varrho_8_p",
                    "rho_8_m",
                    "varrho_8_m",
                    "rho_9_p",
                    "varrho_9_p",
                    "rho_9_m",
                    "varrho_9_m",
                    "rho_10_p",
                    "varrho_10_p",
                    "rho_10_m",
                    "varrho_10_m",
                    "rho_11_p",
                    "varrho_11_p",
                    "rho_11_m",
                    "varrho_11_m",
                    "rho_12_p",
                    "varrho_12_p",
                    "rho_12_m",
                    "varrho_12_m",
                ),
            )

        elif self.use_fourth_moment and self.use_eta:
            self.rho_stats = Table(
                [
                    rho_0.meanr,
                    rho_0.xip,
                    rho_0.varxip,
                    rho_0.xim,
                    rho_0.varxim,
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
                    rho_6.xip,
                    rho_6.varxip,
                    rho_6.xim,
                    rho_6.varxim,
                    rho_7.xip,
                    rho_7.varxip,
                    rho_7.xim,
                    rho_7.varxim,
                    rho_8.xip,
                    rho_8.varxip,
                    rho_8.xim,
                    rho_8.varxim,
                    rho_9.xip,
                    rho_9.varxip,
                    rho_9.xim,
                    rho_9.varxim,
                    rho_10.xip,
                    rho_10.varxip,
                    rho_10.xim,
                    rho_10.varxim,
                    rho_11.xip,
                    rho_11.varxip,
                    rho_11.xim,
                    rho_11.varxim,
                    rho_12.xip,
                    rho_12.varxip,
                    rho_12.xim,
                    rho_12.varxim,
                    rho_13.xip,
                    rho_13.varxip,
                    rho_13.xim,
                    rho_13.varxim,
                    rho_14.xip,
                    rho_14.varxip,
                    rho_14.xim,
                    rho_14.varxim,
                ],
                names=(
                    "theta",
                    "rho_0_p",
                    "varrho_0_p",
                    "rho_0_m",
                    "varrho_0_m",
                    "rho_1_p",
                    "varrho_1_p",
                    "rho_1_m",
                    "varrho_1_m",
                    "rho_2_p",
                    "varrho_2_p",
                    "rho_2_m",
                    "varrho_2_m",
                    "rho_3_p",
                    "varrho_3_p",
                    "rho_3_m",
                    "varrho_3_m",
                    "rho_4_p",
                    "varrho_4_p",
                    "rho_4_m",
                    "varrho_4_m",
                    "rho_5_p",
                    "varrho_5_p",
                    "rho_5_m",
                    "varrho_5_m",
                    "rho_6_p",
                    "varrho_6_p",
                    "rho_6_m",
                    "varrho_6_m",
                    "rho_7_p",
                    "varrho_7_p",
                    "rho_7_m",
                    "varrho_7_m",
                    "rho_8_p",
                    "varrho_8_p",
                    "rho_8_m",
                    "varrho_8_m",
                    "rho_9_p",
                    "varrho_9_p",
                    "rho_9_m",
                    "varrho_9_m",
                    "rho_10_p",
                    "varrho_10_p",
                    "rho_10_m",
                    "varrho_10_m",
                    "rho_11_p",
                    "varrho_11_p",
                    "rho_11_m",
                    "varrho_11_m",
                    "rho_12_p",
                    "varrho_12_p",
                    "rho_12_m",
                    "varrho_12_m",
                    "rho_13_p",
                    "varrho_13_p",
                    "rho_13_m",
                    "varrho_13_m",
                    "rho_14_p",
                    "varrho_14_p",
                    "rho_14_m",
                    "varrho_14_m",
                ),
            )

        else:
            self.rho_stats = Table(
                [
                    rho_0.meanr,
                    rho_0.xip,
                    rho_0.varxip,
                    rho_0.xim,
                    rho_0.varxim,
                    rho_1.xip,
                    rho_1.varxip,
                    rho_1.xim,
                    rho_1.varxim,
                    rho_2.xip,
                    rho_2.varxip,
                    rho_2.xim,
                    rho_2.varxim,
                ],
                names=(
                    "theta",
                    "rho_0_p",
                    "varrho_0_p",
                    "rho_0_m",
                    "varrho_0_m",
                    "rho_1_p",
                    "varrho_1_p",
                    "rho_1_m",
                    "varrho_1_m",
                    "rho_2_p",
                    "varrho_2_p",
                    "rho_2_m",
                    "varrho_2_m",
                ),
            )

        if self.verbose:
            print("Done...")

        if save_cov:
            if self.verbose:
                print("Computing the covariance...")
            rhos = [rho_0, rho_1, rho_2]
            if self.use_eta:
                rhos += [rho_3, rho_4, rho_5]
            if self.use_fourth_moment:
                rhos += [rho_6, rho_7, rho_8, rho_9, rho_10, rho_11, rho_12]
            if self.use_fourth_moment and self.use_eta:
                rhos += [rho_13, rho_14]
            cov = treecorr.estimate_multi_cov(rhos, var_method, func=func)

            use_eta_str = "" if self.use_eta else "no_eta"
            use_fourth_moment_str = (
                "" if not self.use_fourth_moment else "w_fourth_moment"
            )
            np.save(
                self.catalogs._output
                + "/"
                + "cov_rho_"
                + catalog_id
                + use_eta_str
                + use_fourth_moment_str,
                cov,
            )

        self.save_rho_stats(
            filename
        )  # A bit dirty just because of consistency of the datatype
        self.load_rho_stats(filename)

    def save_rho_stats(self, filename):
        self.rho_stats.write(
            self.catalogs._output + "/" + filename,
            format="fits",
            overwrite=True,
        )

    def load_rho_stats(self, filename):
        self.rho_stats = fits.getdata(self.catalogs._output + "/" + filename)

    def save_treecorr(self, treecorr_gg, filename, write_patch_results, write_cov):
        treecorr_gg.write(
            self.catalogs._output + "/" + filename,
            write_patch_results=write_patch_results,
            write_cov=write_cov,
        )

    def plot_rho_stats(
        self,
        filenames,
        colors,
        catalog_ids,
        abs=True,
        offset=0,
        savefig=None,
        legend="each",
        title=None,
        show=False,
        close=True,
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

        offset : float
            Offset to apply to the different versions for better visualisation.

        savefig : str
            If not None, saves the figure with the name given in savefig.

        legend : str, optional
            allowed are "each" (default; legends in each panel);
            "outside" (legend outside of panels); "none" (no legend)

        title : str, optional
            global plot tite, default is ``None``

        show : bool, optional
            If True, display the plot. Default is False.

        close : bool, optional
            If True, close the plot after saving. Default is True.
        """
        e_psf = "e^\mathrm{PSF}"
        delta_e_psf = r"\delta e^\mathrm{PSF}"
        delta_T_psf = r"\delta T^\mathrm{PSF}"

        titles = [
            rf"$\langle {e_psf} {e_psf} \rangle$",
            rf"$\langle {delta_e_psf} {delta_e_psf} \rangle$",
            rf"$\langle {e_psf} {delta_e_psf} \rangle$",
            rf"$\langle {delta_T_psf} {delta_T_psf} \rangle$",
            rf"$\langle {delta_e_psf} {delta_T_psf} \rangle$",
            rf"$\langle {e_psf} {delta_T_psf} \rangle$",
        ]

        # To adapt to the new boolean argument
        fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(15, 9))
        ax = ax.flatten()

        for file_idx, (filename, color, cat_id) in enumerate(
            zip(filenames, colors, catalog_ids)
        ):  # Plot for the different catalogs
            self.load_rho_stats(filename)

            for i in range(6):
                xlabel = r"$\theta$ [arcmin]" if i > 2 else ""

                ylabel = rf"$\rho_{i}(\theta)$"
                label = rf"{cat_id}"

                theta = self.rho_stats["theta"]

                theta_widths = np.diff(theta)
                theta_widths = np.append(theta_widths, theta_widths[-1])

                jitter_fraction = (file_idx - (len(filenames) - 1) / 2) * offset
                jittered_theta = theta + jitter_fraction * theta_widths

                if abs:
                    ax[i].errorbar(
                        jittered_theta,
                        np.abs(self.rho_stats["rho_" + str(i) + "_p"]),
                        yerr=np.sqrt(self.rho_stats["varrho_" + str(i) + "_p"]),
                        label=label,
                        color=color,
                        fmt="o",
                        capsize=2,
                    )
                    ax[i].set_xlabel(xlabel)
                    ax[i].set_ylabel(ylabel)
                    ax[i].set_xscale("log")
                    ax[i].set_yscale("log")
                else:
                    # Plot the negative values of the rho-stats in dashed lines
                    neg_dash(
                        ax[i],
                        jittered_theta,
                        self.rho_stats["rho_" + str(i) + "_p"],
                        yerr_in=np.sqrt(self.rho_stats["varrho_" + str(i) + "_p"]),
                        vertical_lines=False,
                        rho_nb=str(i),
                        cat_id=cat_id,
                        xlabel=xlabel,
                        ylabel=ylabel,
                        semilogx=True,
                        semilogy=True,
                        capsize=2,
                        color=color,
                    )
                ax[i].set_xlim(
                    float(self._treecorr_config["min_sep"]),
                    float(self._treecorr_config["max_sep"]),
                )

                if legend == "each":
                    ax[i].legend(loc="best", fontsize="small")

        for i, title_ in enumerate(titles):
            ax[i].set_title(title_)

        if legend == "outside":
            handles, labels = ax[-1].get_legend_handles_labels()
            fig.legend(
                handles,
                labels,
                loc="upper center",
                bbox_to_anchor=(0.5, 0.0),
                ncol=3,
                frameon=False,
            )

        if title:
            plt.suptitle(title)

        plt.tight_layout()
        if savefig is not None:
            plt.savefig(self.catalogs._output + "/" + savefig, bbox_inches="tight")

        if show:
            plt.show()

        if close:
            plt.close()


class TauStat:
    """
    TauStat

    Class to compute the tau statistics (Gatti 2022) of a PSF and gal catalogue.
    """

    def __init__(
        self,
        params=None,
        output=None,
        use_eta=True,
        scalar_eta=False,
        treecorr_config=None,
        use_fourth_moment=False,
        catalogs=None,
        verbose=False,
    ):
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
                "var_method": "jackknife",
            }
        else:
            self._treecorr_config = treecorr_config

        self.use_eta = use_eta
        self.scalar_eta = scalar_eta
        self.use_fourth_moment = use_fourth_moment
        if self.scalar_eta and not self.use_eta:
            print(
                "Warning: scalar_eta is set to True but use_eta is set to False. Setting use_eta to True."
            )
            self.use_eta = True
        self.verbose = verbose

    def build_cat_to_compute_tau(
        self,
        path_cat,
        cat_type,
        catalog_id="",
        mask=None,
        hdu=1,
    ):
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

        mask : np.array
            A mask array to select only the relevant stars in the catalogue. If None, no mask is applied. (Default: None)

        hdu : int, optional
            HDU number of input FITS file, default is 1
        """

        if cat_type == "psf":
            psf_cat = self.catalogs.read_shear_cat(
                path_gal=None, path_psf=path_cat, hdu=hdu
            )

            if self.verbose:
                print("Building catalogs...")

            self.catalogs.build_catalog(
                cat=psf_cat,
                cat_type="psf",
                key="psf_" + catalog_id,
                mask=mask,
            )
            patch_centers = self.catalogs.catalogs_dict[
                "psf_" + catalog_id
            ].patch_centers
            self.catalogs.build_catalog(
                cat=psf_cat,
                cat_type="psf_error",
                key="psf_error_" + catalog_id,
                patch_centers=patch_centers,
                mask=mask,
            )
            if self.use_eta:
                self.catalogs.build_catalog(
                    cat=psf_cat,
                    cat_type="psf_size_error",
                    key="psf_size_error_" + catalog_id,
                    patch_centers=patch_centers,
                    mask=mask,
                )
            if self.use_fourth_moment:
                self.catalogs.build_catalog(
                    cat=psf_cat,
                    cat_type="psf_fourth_moment",
                    key="psf_fourth_moment_" + catalog_id,
                    patch_centers=patch_centers,
                    mask=mask,
                )
                self.catalogs.build_catalog(
                    cat=psf_cat,
                    cat_type="psf_fourth_moment_error",
                    key="psf_fourth_moment_error_" + catalog_id,
                    patch_centers=patch_centers,
                    mask=mask,
                )
                self.catalogs.build_catalog(
                    cat=psf_cat,
                    cat_type="psf_size_error",
                    key="psf_size_error_" + catalog_id,
                    patch_centers=patch_centers,
                    mask=mask,
                )
            if self.use_fourth_moment:
                self.catalogs.build_catalog(
                    cat=psf_cat,
                    cat_type="psf_fourth_moment",
                    key="psf_fourth_moment_" + catalog_id,
                    patch_centers=patch_centers,
                    mask=mask,
                )
                self.catalogs.build_catalog(
                    cat=psf_cat,
                    cat_type="psf_fourth_moment_error",
                    key="psf_fourth_moment_error_" + catalog_id,
                    patch_centers=patch_centers,
                    mask=mask,
                )

            del psf_cat

        else:
            gal_cat = self.catalogs.read_shear_cat(
                path_gal=path_cat, path_psf=None, hdu=hdu
            )

            if self.verbose:
                print("Building catalog...")

            try:
                patch_centers = self.catalogs.catalogs_dict[
                    "psf_" + catalog_id
                ].patch_centers
            except KeyError:
                warnings.warn("You should build psf catalog before galaxy catalog.")
                patch_centers = None
            self.catalogs.build_catalog(
                cat=gal_cat,
                cat_type="gal",
                key="gal_" + catalog_id,
                mask=mask,
                patch_centers=patch_centers,
            )

            del gal_cat

        if self.verbose:
            print("Catalogs successfully built...")
            self.catalogs.show_catalogs()

    def compute_tau_stats(
        self,
        catalog_id,
        filename,
        save_cov=False,
        func=None,
        var_method="jackknife",
        write_treecorr=False,
        write_patch_results=False,
        write_cov=False,
    ):
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
            print(
                "Computation of the tau statistics of " + catalog_id + " in progress..."
            )

        if write_treecorr:
            filename_no_ext = filename.split(".")[0]

        tau_0 = treecorr.GGCorrelation(self._treecorr_config)
        tau_0.process(
            self.catalogs.get_cat("gal_" + catalog_id),
            self.catalogs.get_cat("psf_" + catalog_id),
        )
        tau_2 = treecorr.GGCorrelation(self._treecorr_config)
        tau_2.process(
            self.catalogs.get_cat("gal_" + catalog_id),
            self.catalogs.get_cat("psf_error_" + catalog_id),
        )

        if write_treecorr:
            self.save_treecorr(
                tau_0, filename_no_ext + "_tau_0.txt", write_patch_results, write_cov
            )
            self.save_treecorr(
                tau_2, filename_no_ext + "_tau_2.txt", write_patch_results, write_cov
            )

        if self.use_eta:
            tau_5 = treecorr.GGCorrelation(self._treecorr_config)
            tau_5.process(
                self.catalogs.get_cat("gal_" + catalog_id),
                self.catalogs.get_cat("psf_size_error_" + catalog_id),
            )
            if write_treecorr:
                self.save_treecorr(
                    tau_5,
                    filename_no_ext + "_tau_5.txt",
                    write_patch_results,
                    write_cov,
                )

        if self.use_fourth_moment:
            tau_6 = treecorr.GGCorrelation(self._treecorr_config)
            tau_6.process(
                self.catalogs.get_cat("gal_" + catalog_id),
                self.catalogs.get_cat("psf_fourth_moment_" + catalog_id),
            )
            tau_7 = treecorr.GGCorrelation(self._treecorr_config)
            tau_7.process(
                self.catalogs.get_cat("gal_" + catalog_id),
                self.catalogs.get_cat("psf_fourth_moment_error_" + catalog_id),
            )
            if write_treecorr:
                self.save_treecorr(
                    tau_6,
                    filename_no_ext + "_tau_6.txt",
                    write_patch_results,
                    write_cov,
                )
                self.save_treecorr(
                    tau_7,
                    filename_no_ext + "_tau_7.txt",
                    write_patch_results,
                    write_cov,
                )

        if self.use_eta and not self.use_fourth_moment:
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
                    "theta",
                    "tau_0_p",
                    "vartau_0_p",
                    "tau_0_m",
                    "vartau_0_m",
                    "tau_2_p",
                    "vartau_2_p",
                    "tau_2_m",
                    "vartau_2_m",
                    "tau_5_p",
                    "vartau_5_p",
                    "tau_5_m",
                    "vartau_5_m",
                ),
            )

        elif self.use_fourth_moment and not self.use_eta:
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
                    tau_6.xip,
                    tau_6.varxip,
                    tau_6.xim,
                    tau_6.varxim,
                    tau_7.xip,
                    tau_7.varxip,
                    tau_7.xim,
                    tau_7.varxim,
                ],
                names=(
                    "theta",
                    "tau_0_p",
                    "vartau_0_p",
                    "tau_0_m",
                    "vartau_0_m",
                    "tau_2_p",
                    "vartau_2_p",
                    "tau_2_m",
                    "vartau_2_m",
                    "tau_6_p",
                    "vartau_6_p",
                    "tau_6_m",
                    "vartau_6_m",
                    "tau_7_p",
                    "vartau_7_p",
                    "tau_7_m",
                    "vartau_7_m",
                ),
            )

        elif self.use_eta and self.use_fourth_moment:
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
                    tau_6.xip,
                    tau_6.varxip,
                    tau_6.xim,
                    tau_6.varxim,
                    tau_7.xip,
                    tau_7.varxip,
                    tau_7.xim,
                    tau_7.varxim,
                ],
                names=(
                    "theta",
                    "tau_0_p",
                    "vartau_0_p",
                    "tau_0_m",
                    "vartau_0_m",
                    "tau_2_p",
                    "vartau_2_p",
                    "tau_2_m",
                    "vartau_2_m",
                    "tau_5_p",
                    "vartau_5_p",
                    "tau_5_m",
                    "vartau_5_m",
                    "tau_6_p",
                    "vartau_6_p",
                    "tau_6_m",
                    "vartau_6_m",
                    "tau_7_p",
                    "vartau_7_p",
                    "tau_7_m",
                    "vartau_7_m",
                ),
            )

        else:
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
                ],
                names=(
                    "theta",
                    "tau_0_p",
                    "vartau_0_p",
                    "tau_0_m",
                    "vartau_0_m",
                    "tau_2_p",
                    "vartau_2_p",
                    "tau_2_m",
                    "vartau_2_m",
                ),
            )

        if self.verbose:
            print("Done...")

        if save_cov:
            if self.verbose:
                print("Computing the covariance...")
            taus = [tau_0, tau_2]
            if self.use_eta:
                taus += [tau_5]
            if self.use_fourth_moment:
                taus += [tau_6, tau_7]
            cov = treecorr.estimate_multi_cov(taus, var_method, func=func)

            use_eta_str = "" if self.use_eta else "no_eta"
            use_fourth_moment_str = (
                "" if not self.use_fourth_moment else "w_fourth_moment"
            )
            np.save(
                self.catalogs._output
                + "/"
                + "cov_tau_"
                + catalog_id
                + use_eta_str
                + use_fourth_moment_str,
                cov,
            )

        self.save_tau_stats(
            filename
        )  # A bit dirty just because of consistency of the datatype :/
        self.load_tau_stats(filename)

    def save_tau_stats(self, filename):
        self.tau_stats.write(
            self.catalogs._output + "/" + filename,
            format="fits",
            overwrite=True,
        )

    def load_tau_stats(self, filename):
        self.tau_stats = fits.getdata(self.catalogs._output + "/" + filename)

    def save_treecorr(self, treecorr_gg, filename, write_patch_results, write_cov):
        treecorr_gg.write(
            self.catalogs._output + "/" + filename,
            write_patch_results=write_patch_results,
            write_cov=write_cov,
        )

    def plot_tau_stats(
        self,
        filenames,
        colors,
        catalog_ids,
        cov_paths=None,
        offset=0,
        savefig=None,
        plot_tau_m=True,
        plot_theta_times_tau=True,
        legend="inside",
        show=False,
        close=True,
        fmt="",
        capsize=2,
    ):
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

        cov_paths : list str
            The paths to the covariance to use. If None the covariance saved in the tau-stats object is used.

        offset : float
            Offset to apply to the different versions for better visualisation.

        savefig : str
            If not None, saves the figure with the name given in savefig.

        plot_tau_m : bool
            If True, plot the tau - additionally.

        plot_theta_times_tau : bool
            If True, plot theta x tau for the 2 and 5 components.

        legend : str, optional
            allowed are "each" (default; legends in each panel), "outside" (legend outside of panels)

        show : bool, optional
            If True, displays the plot. Default is False.

        close : bool, optional
            If True, closes the plot after saving/showing. Default is True.

        fmt : str, optional
            Format string for the plot. Default is ''.

        capsize : int, optional
            Cap size for error bars. Default is 2.

        Return
        ------
        fig : Figure

        ax : Axes
        """
        e_obs = "e^\mathrm{obs}"
        e_psf = "e^\mathrm{PSF}"
        delta_e_psf = "\delta e^\mathrm{PSF}"
        delta_T_psf = "\delta T^\mathrm{PSF}"

        titles = [
            rf"$\langle {e_obs} {e_psf} \rangle$",
            rf"$\langle {e_obs} {delta_e_psf} \rangle$",
            rf"$\langle {e_obs} {delta_T_psf} \rangle$",
        ]

        dict_index_tau = {
            0: "0",
            1: "2",
            2: "5",
        }
        # To adapt to the new boolean fields
        nrows = 1 + plot_tau_m

        fig, ax = plt.subplots(nrows=nrows, ncols=3, figsize=(15, 6))

        if cov_paths is not None:
            print("Using the error bars from the covariance files for the + part.")
        else:
            print("Using the error bars from the tau-statistics files for the + part.")

        if nrows == 1:
            ax = ax.reshape(1, 3)

        for i in range(3):
            for j in range(nrows):
                iterator_ = (
                    zip(filenames, colors, catalog_ids)
                    if cov_paths is None
                    else zip(filenames, colors, catalog_ids, cov_paths)
                )
                for file_idx, item in enumerate(
                    iterator_
                ):  # Plot for the different catalogs
                    if cov_paths is None:
                        filename, color, cat_id = item
                    else:
                        filename, color, cat_id, cov_path = item
                        cov_tau = np.load(self.catalogs._output + "/" + cov_path)

                    self.load_tau_stats(filename)
                    if cov_paths is not None:
                        if cov_tau.shape[0] != self.tau_stats["theta"].shape[0] * 3:
                            raise ValueError(
                                f"The covariance file {cov_path} has a different number of theta points\n"
                                f"than the tau statistics file {filename}. Please check your files.\n"
                                f"Covariance shape: {cov_tau.shape}, Tau shape: {self.tau_stats['theta'].shape}"
                            )
                        # Check if the covariance is a square matrix
                        if cov_tau.shape[0] != cov_tau.shape[1] or cov_tau.ndim != 2:
                            raise ValueError(
                                f"The covariance file {cov_path} is not a square matrix."
                            )

                    p_or_m = "m" if j else "p"
                    p_or_m_label = "-" if j else "+"
                    xlabel = r"$\theta$ [arcmin]" if (j == nrows - 1) else ""
                    ylabel = (
                        rf"$\tau_{{{dict_index_tau[i]}, {p_or_m_label}}}(\theta)$"
                        if (i == 0) or not plot_theta_times_tau
                        else rf"$\tau_{{{dict_index_tau[i]}, {p_or_m_label}}}(\theta)\theta$"
                    )
                    label = rf"{cat_id}"
                    factor_theta = (
                        np.ones_like(self.tau_stats["theta"])
                        if (i == 0) or not plot_theta_times_tau
                        else self.tau_stats["theta"]
                    )
                    y = (
                        self.tau_stats["tau_" + dict_index_tau[i] + "_" + p_or_m]
                        * factor_theta
                    )
                    if cov_path is None or p_or_m == "m":
                        cov_diag = self.tau_stats[
                            "vartau_" + dict_index_tau[i] + "_" + p_or_m
                        ]
                    else:
                        num_theta_bins = factor_theta.shape[0]
                        cov_diag = np.diag(
                            cov_tau[
                                i * num_theta_bins : (i + 1) * num_theta_bins,
                                i * num_theta_bins : (i + 1) * num_theta_bins,
                            ]
                        )

                    yerr_in = np.sqrt(cov_diag) * factor_theta

                    theta = self.tau_stats["theta"]

                    theta_widths = np.diff(theta)
                    theta_widths = np.append(theta_widths, theta_widths[-1])

                    jitter_fraction = (file_idx - (len(filenames) - 1) / 2) * offset
                    jittered_theta = theta + jitter_fraction * theta_widths

                    ax[j, i].errorbar(
                        jittered_theta,
                        y,
                        yerr=yerr_in,
                        label=label,
                        color=color,
                        fmt=fmt,
                        capsize=capsize,
                    )
                ax[j, i].set_xlim(
                    self._treecorr_config["min_sep"],
                    self._treecorr_config["max_sep"],
                )
                ax[j, i].set_xlabel(xlabel)
                ax[j, i].set_xscale("log")
                # Recover the text offset
                fig.canvas.draw()
                text_offset = ax[j, i].yaxis.get_offset_text().get_text()
                ylabel += text_offset
                ax[j, i].yaxis.get_offset_text().set_visible(False)

                ax[j, i].set_ylabel(ylabel)
                if legend == "inside":
                    ax[j, i].legend(loc="best", fontsize="small")

                ax[j, i].set_title(titles[i])

        if legend == "outside":
            handles, labels = ax[-1, -1].get_legend_handles_labels()
            fig.legend(
                handles,
                labels,
                loc="upper center",
                bbox_to_anchor=(0.5, 0.0),
                ncol=3,
                frameon=False,
            )

        plt.tight_layout()
        if savefig is not None:
            plt.savefig(self.catalogs._output + "/" + savefig, bbox_inches="tight")

        if show:
            plt.show()

        if close:
            plt.close()

        return fig, ax


class PSFErrorFit:
    """
    PSFErrorFit

    This class uses rho and tau statistics data to fit parameters alpha, beta and eta in the psf error model
    (Gatti et al. 2022). It includes methods to plot the best fit value and the tau statistics as well as
    as the systematic error for different catalogs.
    A Likelihood-Based Inference is used assuming a Gaussian likelihood.
    """

    def __init__(
        self,
        rho_stat_handler,
        tau_stat_handler,
        data_directory,
        use_eta=True,
        use_fourth_moment=False,
    ):
        self.rho_stat_handler = rho_stat_handler
        self.tau_stat_handler = tau_stat_handler
        print(
            "Class created. Don't forget to load your rho and tau statistics and define your prior and your likelihood."
        )
        self.cov_rho = None
        self.cov_tau = None
        self.init_log_prior()

        self.use_eta = use_eta
        self.use_fourth_moment = use_fourth_moment

        def log_likelihood(theta, y, inv_cov):
            y_model = self.model(theta, self.use_eta, self.use_fourth_moment)
            d = y_model - y
            return -0.5 * d.T @ inv_cov @ d

        self.log_likelihood = log_likelihood
        self.data_directory = data_directory

        self.rho_stat_handler.catalogs._output = (
            self.data_directory
        )  # Change the path to the specified data directory
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
        self.rho_stat_handler.catalogs._output = (
            self.data_directory
        )  # Change the path to the specified data directory
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

    def load_covariance(self, filename, cov_type="rho"):
        """
        load_covariance

        Load a covariance matrix to run the inference.

        Parameters
        ----------
        filename : str
            Name of the file containing the covariance matrix.
        """
        if cov_type == "rho":
            self.cov_rho = np.load(self.data_directory + "/" + filename)
            # Reshape the covariance if needed
            nbins = self.rho_stat_handler.rho_stats["theta"].shape[0]
            # Check shape
            if self.use_eta and self.use_fourth_moment:
                target_shape = 15 * nbins
            elif self.use_eta and (not self.use_fourth_moment):
                target_shape = 6 * nbins
            elif (not self.use_eta) and self.use_fourth_moment:
                target_shape = 10 * nbins
            else:
                target_shape = 3 * nbins
            assert self.cov_rho.shape[0] == target_shape, (
                "The shape of the covariance matrix is not correct."
            )
        else:
            self.cov_tau = np.load(self.data_directory + "/" + filename)
            nbins = self.tau_stat_handler.tau_stats["theta"].shape[0]
            if not self.use_eta:
                self.cov_tau = self.cov_tau[: 2 * nbins, : 2 * nbins]
            # Check shape
            target_shape = 3 * nbins if self.use_eta else 2 * nbins
            assert self.cov_tau.shape[0] == target_shape, (
                "The shape of the covariance matrix is not correct."
            )

    def init_log_prior(
        self,
        low_alpha=-2.0,
        high_alpha=2.0,
        low_beta=-10.0,
        high_beta=10.0,
        low_eta=-20.0,
        high_eta=20.0,
        low_alpha_4=-20.0,
        high_alpha_4=20.0,
        low_beta_4=-20.0,
        high_beta_4=20.0,
    ):
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

        low_alpha_4 : float
            Lower bound on alpha_4 prior (Default: -20.0).

        high_alpha_4 : float
            Upper bound on alpha_4 prior (Default: 20.0).

        low_beta_4 : float
            Lower bound on beta_4 prior (Default: -20.0).

        high_beta_4 : float
            Upper bound on beta_4 prior (Default: 20.0).

        Returns
        -------
        function
            Store in the attribute `log_prior` of the class a function that returns the value of the log_prior
            given a set of parameters theta.
        """

        def log_prior(theta):
            alpha, beta, eta, alpha_4, beta_4 = theta
            if (
                low_alpha <= alpha <= high_alpha
                and low_beta <= beta <= high_beta
                and low_eta <= eta <= high_eta
                and low_alpha_4 <= alpha_4 <= high_alpha_4
                and low_beta_4 <= beta_4 <= high_beta_4
            ):
                return 0.0
            return -np.inf

        self.log_prior = log_prior

    def model(self, theta, use_eta, use_fourth_moment):
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
        alpha, beta, eta, alpha_4, beta_4 = theta

        rhos = self.rho_stat_handler.rho_stats
        if (not use_eta) and (not use_fourth_moment):
            tau_0_p = alpha * rhos["rho_0_p"] + beta * rhos["rho_2_p"]
            tau_2_p = alpha * rhos["rho_2_p"] + beta * rhos["rho_1_p"]
        elif (use_eta) and (not use_fourth_moment):
            tau_0_p = (
                alpha * rhos["rho_0_p"] + beta * rhos["rho_2_p"] + eta * rhos["rho_5_p"]
            )
            tau_2_p = (
                alpha * rhos["rho_2_p"] + beta * rhos["rho_1_p"] + eta * rhos["rho_4_p"]
            )
            tau_5_p = (
                alpha * rhos["rho_5_p"] + beta * rhos["rho_4_p"] + eta * rhos["rho_3_p"]
            )
        elif (not use_eta) and (use_fourth_moment):
            tau_0_p = (
                alpha * rhos["rho_0_p"]
                + beta * rhos["rho_2_p"]
                + alpha_4 * rhos["rho_7_p"]
                + beta_4 * rhos["rho_12_p"]
            )
            tau_2_p = (
                alpha * rhos["rho_2_p"]
                + beta * rhos["rho_1_p"]
                + alpha_4 * rhos["rho_8_p"]
                + beta_4 * rhos["rho_11_p"]
            )
            tau_6_p = (
                alpha * rhos["rho_7_p"]
                + beta * rhos["rho_8_p"]
                + alpha_4 * rhos["rho_6_p"]
                + beta_4 * rhos["rho_10_p"]
            )
            tau_7_p = (
                alpha * rhos["rho_12_p"]
                + beta * rhos["rho_11_p"]
                + alpha_4 * rhos["rho_10_p"]
                + beta_4 * rhos["rho_9_p"]
            )
        elif (use_eta) and (use_fourth_moment):
            tau_0_p = (
                alpha * rhos["rho_0_p"]
                + beta * rhos["rho_2_p"]
                + eta * rhos["rho_5_p"]
                + alpha_4 * rhos["rho_7_p"]
                + beta_4 * rhos["rho_12_p"]
            )
            tau_2_p = (
                alpha * rhos["rho_2_p"]
                + beta * rhos["rho_1_p"]
                + eta * rhos["rho_4_p"]
                + alpha_4 * rhos["rho_8_p"]
                + beta_4 * rhos["rho_11_p"]
            )
            tau_5_p = (
                alpha * rhos["rho_5_p"]
                + beta * rhos["rho_4_p"]
                + eta * rhos["rho_3_p"]
                + alpha_4 * rhos["rho_13_p"]
                + beta_4 * rhos["rho_14_p"]
            )
            tau_6_p = (
                alpha * rhos["rho_7_p"]
                + beta * rhos["rho_8_p"]
                + eta * rhos["rho_13_p"]
                + alpha_4 * rhos["rho_6_p"]
                + beta_4 * rhos["rho_10_p"]
            )
            tau_7_p = (
                alpha * rhos["rho_12_p"]
                + beta * rhos["rho_11_p"]
                + eta * rhos["rho_14_p"]
                + alpha_4 * rhos["rho_10_p"]
                + beta_4 * rhos["rho_9_p"]
            )

        model_output = [tau_0_p, tau_2_p]

        if use_eta:
            model_output += [tau_5_p]
        if use_fourth_moment:
            model_output += [tau_6_p, tau_7_p]

        model_output = np.array(model_output)

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
        log_posterior = lp + self.log_likelihood(theta, y, inv_cov)
        return log_posterior

    def run_chain(
        self,
        init=np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
        nwalkers=124,
        nsamples=10000,
        discard=300,
        thin=100,
        verbose=True,
        savefig=None,
        npatch=200,
        apply_debias=False,
    ):
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
        if (not self.use_eta) and (not self.use_fourth_moment):
            ndim = 2
            dim_to_use = [0, 1]
        elif (not self.use_eta) and (self.use_fourth_moment):
            ndim = 4
            dim_to_use = [0, 1, 3, 4]
        elif (self.use_eta) and (not self.use_fourth_moment):
            ndim = 3
            dim_to_use = [0, 1, 2]
        elif (self.use_eta) and (self.use_fourth_moment):
            ndim = 5
            dim_to_use = [0, 1, 2, 3, 4]
        assert self.rho_stat_handler.rho_stats is not None, (
            "Please load rho statistics data."
        )  # Check if data was loaded
        assert self.tau_stat_handler.tau_stats is not None, (
            "Please load tau statistics data."
        )
        # assert (np.all(self.rho_stat_handler.rho_stats["theta"] == self.tau_stat_handler.tau_stats["theta"])), ("The rho and tau statistics have not the same angular scales. Check that they come from the same catalog with the same treecorr config.")
        # Check that the abssiss are the same

        assert self.cov_tau is not None, "Please load a covariance matrix"

        if (not self.use_eta) and (not self.use_fourth_moment):
            assert (
                self.cov_tau.shape[0]
                == 2 * self.rho_stat_handler.rho_stats["theta"].shape[0]
            ), (
                f"The covariance matrix does not have the right shape. Shape: {self.cov_tau.shape}"
            )
        elif (not self.use_eta) and (self.use_fourth_moment):
            assert (
                self.cov_tau.shape[0]
                == 4 * self.rho_stat_handler.rho_stats["theta"].shape[0]
            ), (
                f"The covariance matrix does not have the right shape. Shape: {self.cov_tau.shape}"
            )
        elif (self.use_eta) and (not self.use_fourth_moment):
            assert (
                self.cov_tau.shape[0]
                == 3 * self.rho_stat_handler.rho_stats["theta"].shape[0]
            ), (
                f"The covariance matrix does not have the right shape. Shape: {self.cov_tau.shape}"
            )
        elif (self.use_eta) and (self.use_fourth_moment):
            assert (
                self.cov_tau.shape[0]
                == 5 * self.rho_stat_handler.rho_stats["theta"].shape[0]
            ), (
                f"The covariance matrix does not have the right shape. Shape: {self.cov_tau.shape}"
            )

        inv_cov = np.linalg.inv(self.cov_tau)

        output = [
            self.tau_stat_handler.tau_stats["tau_0_p"],
            self.tau_stat_handler.tau_stats["tau_2_p"],
        ]
        if self.use_eta:
            output += [self.tau_stat_handler.tau_stats["tau_5_p"]]
        if self.use_fourth_moment:
            output += [
                self.tau_stat_handler.tau_stats["tau_6_p"],
                self.tau_stat_handler.tau_stats["tau_7_p"],
            ]

        output = np.array(output).flatten()

        if apply_debias:
            inv_cov = (npatch - output.shape[0] - 2) / (npatch - 1) * inv_cov

        init = init + 1e-1 * np.random.randn(nwalkers, 5)

        sampler = emcee.EnsembleSampler(
            nwalkers, 5, self.log_probability, args=(output, inv_cov)
        )

        print("Run MCMC analysis...")
        sampler.run_mcmc(init, nsamples, progress=verbose)
        print("Done")

        tau = sampler.get_autocorr_time(quiet=True)

        if verbose:
            print("Autocorrelation-time:")
            print(tau)

        labels = [r"$\alpha$", r"$\beta$"]
        if self.use_eta:
            labels += [r"$\eta$"]
        if self.use_fourth_moment:
            labels += [r"$\alpha_4$", r"$\beta_4$"]

        if savefig is not None:
            fig, axes = plt.subplots(
                ndim, figsize=(10, 7), sharex=True
            )  # Result completely unconstrained. have another look at the covariance matrix
            samples = sampler.get_chain()
            for i in range(ndim):
                ax = axes[i]
                ax.plot(samples[:, :, i], "k", alpha=0.3)
                ax.set_xlim(0, len(samples))
                ax.set_ylabel(labels[i])
                ax.yaxis.set_label_coords(-0.1, 0.5)

                axes[-1].set_xlabel("step number")

            plt.savefig(self.data_directory + "/" + savefig)

        flat_samples = sampler.get_chain(discard=discard, thin=thin, flat=True)
        mcmc_result, q = self.get_mcmc_from_samples(flat_samples)

        # mcmc_result = np.percentile(flat_samples, [16, 50, 84], axis=0)
        # q = np.diff(mcmc_result, axis=0)
        if verbose:
            print(f"Number of samples: {flat_samples.shape[0]}\n")

            print("Parameters constraints")
            print("----------------------")
            for i, param in enumerate(dim_to_use):
                print(
                    "Parameter: "
                    + labels[param]
                    + f"={mcmc_result[1, param]:.4f}^+{q[0, param]:.4f}_{q[1, param]:.4f}"
                )

            print(
                f"Max log_likelihood: {self.log_likelihood(mcmc_result[1, :], output, inv_cov)}"
            )

        flat_samples = flat_samples[:, dim_to_use]
        mcmc_result = mcmc_result[:, dim_to_use]
        q = q[:, dim_to_use]

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
        n_thetas = len(self.rho_stat_handler.rho_stats["theta"])  # number of bins
        if (self.use_eta) and (not self.use_fourth_moment):
            rho_matrix = np.zeros((3 * n_thetas, 3))
        elif (self.use_eta) and (self.use_fourth_moment):
            rho_matrix = np.zeros((5 * n_thetas, 5))
        elif (not self.use_eta) and (self.use_fourth_moment):
            rho_matrix = np.zeros((4 * n_thetas, 4))
        elif (not self.use_eta) and (not self.use_fourth_moment):
            rho_matrix = np.zeros((2 * n_thetas, 2))
        if rho is None:
            rho_stats = self.rho_stat_handler.rho_stats
            for i in range(n_thetas):
                if self.use_eta:
                    rho_matrix[i] = [
                        rho_stats["rho_0_p"][i],
                        rho_stats["rho_2_p"][i],
                        rho_stats["rho_5_p"][i],
                    ]
                    rho_matrix[i + n_thetas] = [
                        rho_stats["rho_2_p"][i],
                        rho_stats["rho_1_p"][i],
                        rho_stats["rho_4_p"][i],
                    ]
                    rho_matrix[i + 2 * n_thetas] = [
                        rho_stats["rho_5_p"][i],
                        rho_stats["rho_4_p"][i],
                        rho_stats["rho_3_p"][i],
                    ]
                else:
                    rho_matrix[i] = [rho_stats["rho_0_p"][i], rho_stats["rho_2_p"][i]]
                    rho_matrix[i + n_thetas] = [
                        rho_stats["rho_2_p"][i],
                        rho_stats["rho_1_p"][i],
                    ]
                if (self.use_eta) and (not self.use_fourth_moment):
                    rho_matrix[i] = [
                        rho_stats["rho_0_p"][i],
                        rho_stats["rho_2_p"][i],
                        rho_stats["rho_5_p"][i],
                    ]
                    rho_matrix[i + n_thetas] = [
                        rho_stats["rho_2_p"][i],
                        rho_stats["rho_1_p"][i],
                        rho_stats["rho_4_p"][i],
                    ]
                    rho_matrix[i + 2 * n_thetas] = [
                        rho_stats["rho_5_p"][i],
                        rho_stats["rho_4_p"][i],
                        rho_stats["rho_3_p"][i],
                    ]
                elif (self.use_eta) and (self.use_fourth_moment):
                    rho_matrix[i] = [
                        rho_stats["rho_0_p"][i],
                        rho_stats["rho_2_p"][i],
                        rho_stats["rho_5_p"][i],
                        rho_stats["rho_7_p"][i],
                        rho_stats["rho_12_p"][i],
                    ]
                    rho_matrix[i + n_thetas] = [
                        rho_stats["rho_2_p"][i],
                        rho_stats["rho_1_p"][i],
                        rho_stats["rho_4_p"][i],
                        rho_stats["rho_8_p"][i],
                        rho_stats["rho_11_p"][i],
                    ]
                    rho_matrix[i + 2 * n_thetas] = [
                        rho_stats["rho_5_p"][i],
                        rho_stats["rho_4_p"][i],
                        rho_stats["rho_3_p"][i],
                        rho_stats["rho_13_p"][i],
                        rho_stats["rho_14_p"][i],
                    ]
                    rho_matrix[i + 3 * n_thetas] = [
                        rho_stats["rho_7_p"][i],
                        rho_stats["rho_8_p"][i],
                        rho_stats["rho_13_p"][i],
                        rho_stats["rho_6_p"][i],
                        rho_stats["rho_10_p"][i],
                    ]
                    rho_matrix[i + 4 * n_thetas] = [
                        rho_stats["rho_12_p"][i],
                        rho_stats["rho_11_p"][i],
                        rho_stats["rho_14_p"][i],
                        rho_stats["rho_10_p"][i],
                        rho_stats["rho_9_p"][i],
                    ]
                elif (not self.use_eta) and (self.use_fourth_moment):
                    rho_matrix[i] = [
                        rho_stats["rho_0_p"][i],
                        rho_stats["rho_2_p"][i],
                        rho_stats["rho_7_p"][i],
                        rho_stats["rho_12_p"][i],
                    ]
                    rho_matrix[i + n_thetas] = [
                        rho_stats["rho_2_p"][i],
                        rho_stats["rho_1_p"][i],
                        rho_stats["rho_8_p"][i],
                        rho_stats["rho_11_p"][i],
                    ]
                    rho_matrix[i + 2 * n_thetas] = [
                        rho_stats["rho_7_p"][i],
                        rho_stats["rho_8_p"][i],
                        rho_stats["rho_6_p"][i],
                        rho_stats["rho_10_p"][i],
                    ]
                    rho_matrix[i + 3 * n_thetas] = [
                        rho_stats["rho_12_p"][i],
                        rho_stats["rho_11_p"][i],
                        rho_stats["rho_10_p"][i],
                        rho_stats["rho_9_p"][i],
                    ]
                elif (not self.use_eta) and (not self.use_fourth_moment):
                    rho_matrix[i] = [rho_stats["rho_0_p"][i], rho_stats["rho_2_p"][i]]
                    rho_matrix[i + n_thetas] = [
                        rho_stats["rho_2_p"][i],
                        rho_stats["rho_1_p"][i],
                    ]
        else:
            rho_stats = rho
            for i in range(n_thetas):
                if (self.use_eta) and (not self.use_fourth_moment):
                    rho_matrix[i] = [rho_stats[0, i], rho_stats[2, i], rho_stats[5, i]]
                    rho_matrix[i + n_thetas] = [
                        rho_stats[2, i],
                        rho_stats[1, i],
                        rho_stats[4, i],
                    ]
                    rho_matrix[i + 2 * n_thetas] = [
                        rho_stats[5, i],
                        rho_stats[4, i],
                        rho_stats[3, i],
                    ]
                elif (self.use_eta) and (self.use_fourth_moment):
                    rho_matrix[i] = [
                        rho_stats[0, i],
                        rho_stats[2, i],
                        rho_stats[5, i],
                        rho_stats[7, i],
                        rho_stats[12, i],
                    ]
                    rho_matrix[i + n_thetas] = [
                        rho_stats[2, i],
                        rho_stats[1, i],
                        rho_stats[4, i],
                        rho_stats[8, i],
                        rho_stats[11, i],
                    ]
                    rho_matrix[i + 2 * n_thetas] = [
                        rho_stats[5, i],
                        rho_stats[4, i],
                        rho_stats[3, i],
                        rho_stats[13, i],
                        rho_stats[14, i],
                    ]
                    rho_matrix[i + 3 * n_thetas] = [
                        rho_stats[7, i],
                        rho_stats[8, i],
                        rho_stats[13, i],
                        rho_stats[6, i],
                        rho_stats[10, i],
                    ]
                    rho_matrix[i + 4 * n_thetas] = [
                        rho_stats[12, i],
                        rho_stats[11, i],
                        rho_stats[14, i],
                        rho_stats[10, i],
                        rho_stats[9, i],
                    ]
                elif (not self.use_eta) and (self.use_fourth_moment):
                    rho_matrix[i] = [
                        rho_stats[0, i],
                        rho_stats[2, i],
                        rho_stats[4, i],
                        rho_stats[9, i],
                    ]
                    rho_matrix[i + n_thetas] = [
                        rho_stats[2, i],
                        rho_stats[1, i],
                        rho_stats[5, i],
                        rho_stats[8, i],
                    ]
                    rho_matrix[i + 2 * n_thetas] = [
                        rho_stats[4, i],
                        rho_stats[5, i],
                        rho_stats[3, i],
                        rho_stats[7, i],
                    ]
                    rho_matrix[i + 3 * n_thetas] = [
                        rho_stats[9, i],
                        rho_stats[8, i],
                        rho_stats[7, i],
                        rho_stats[6, i],
                    ]
                elif (not self.use_eta) and (not self.use_fourth_moment):
                    rho_matrix[i] = [rho_stats[0, i], rho_stats[2, i]]
                    rho_matrix[i + n_thetas] = [
                        rho_stats[2, i],
                        rho_stats[1, i],
                    ]
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
            tau_vec = [tau_stats["tau_0_p"], tau_stats["tau_2_p"]]
            if self.use_eta:
                tau_vec += [tau_stats["tau_5_p"]]
            if self.use_fourth_moment:
                tau_vec += [tau_stats["tau_6_p"], tau_stats["tau_7_p"]]
            tau_vec = np.array(tau_vec).flatten()
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
        assert self.cov_tau is not None, (
            "Please load a covariance matrix for the tau statistics."
        )
        inv_cov = np.linalg.inv(self.cov_tau)
        if apply_debias:
            inv_cov = (npatch - tau_vec.shape[0] - 2) / (npatch - 1) * inv_cov
        return (
            np.linalg.inv(rho_matrix.T @ inv_cov @ rho_matrix)
            @ rho_matrix.T
            @ inv_cov
            @ tau_vec
        )

    def get_least_squares_params_samples(
        self, npatch, apply_debias=False, n_samples=10000, verbose=True
    ):
        """
        Computes the least square optimum of the residuals by sampling rho and tau statistics from their covariance.

        Parameters
        ----------
        npatch : int
            The number of patches used to compute the covariance.
        apply_debias : bool
            If True, apply some debiasing of the inverse of the covariance matrix. (Default: False)
        nsamples : int
            The number of samples to draw from the covariance matrix.
        verbose : bool
            If True, prints several informations.

        Returns
        -------
        np.array
            Samples obtained from the Least-Squares analysis
        np.array
            Best-fit values of the parameters
        np.array
            Error bars at the 68% confidence level.
        """
        assert self.cov_tau is not None, (
            "Please load a covariance matrix for the tau statistics."
        )
        assert self.cov_rho is not None, (
            "Please load a covariance matrix for the rho statistics."
        )
        rho_stats = self.rho_stat_handler.rho_stats
        rho_mean = [rho_stats["rho_0_p"], rho_stats["rho_1_p"], rho_stats["rho_2_p"]]
        if self.use_eta:
            rho_mean += [
                rho_stats["rho_3_p"],
                rho_stats["rho_4_p"],
                rho_stats["rho_5_p"],
            ]
        if self.use_fourth_moment:
            rho_mean += [
                rho_stats["rho_6_p"],
                rho_stats["rho_7_p"],
                rho_stats["rho_8_p"],
                rho_stats["rho_9_p"],
                rho_stats["rho_10_p"],
                rho_stats["rho_11_p"],
                rho_stats["rho_12_p"],
            ]
        if self.use_eta and self.use_fourth_moment:
            rho_mean += [rho_stats["rho_13_p"], rho_stats["rho_14_p"]]
        rho_mean = np.array(rho_mean).flatten()
        tau_stats = self.tau_stat_handler.tau_stats
        tau_mean = [tau_stats["tau_0_p"], tau_stats["tau_2_p"]]
        if self.use_eta:
            tau_mean += [tau_stats["tau_5_p"]]
        if self.use_fourth_moment:
            tau_mean += [tau_stats["tau_6_p"], tau_stats["tau_7_p"]]
        tau_mean = np.array(tau_mean).flatten()

        for i in tqdm(range(n_samples)):
            rho = np.random.multivariate_normal(rho_mean, self.cov_rho)
            if self.use_eta and not self.use_fourth_moment:
                rho = rho.reshape((6, -1))
            elif not self.use_eta and self.use_fourth_moment:
                rho = rho.reshape((10, -1))
            elif self.use_eta and self.use_fourth_moment:
                rho = rho.reshape((15, -1))
            else:
                rho = rho.reshape((3, -1))
            rho_matrix = self.build_rho_matrix(rho)
            tau = np.random.multivariate_normal(tau_mean, self.cov_tau)
            tau_vec = self.build_tau_vec(tau)
            inv_cov = np.linalg.inv(self.cov_tau)
            if apply_debias:
                inv_cov = (npatch - tau_vec.shape[0] - 2) / (npatch - 1) * inv_cov
            if i == 0:
                samples = (
                    np.linalg.inv(rho_matrix.T @ inv_cov @ rho_matrix)
                    @ rho_matrix.T
                    @ inv_cov
                    @ tau_vec
                )
            else:
                samples = np.vstack(
                    (
                        samples,
                        np.linalg.inv(rho_matrix.T @ inv_cov @ rho_matrix)
                        @ rho_matrix.T
                        @ inv_cov
                        @ tau_vec,
                    )
                )
        result, q = self.get_mcmc_from_samples(samples)

        if verbose:
            labels = [r"$\alpha$", r"$\beta$"]
            if self.use_eta:
                labels += [r"$\eta$"]
            if self.use_fourth_moment:
                labels += [r"$\alpha_4$", r"$\beta_4$"]
            print(f"Number of samples: {samples.shape[0]}\n")

            print("Parameters constraints")
            print("----------------------")
            for i in range(2 + self.use_eta + 2 * self.use_fourth_moment):
                print(
                    "Parameter: "
                    + labels[i]
                    + f"={result[1, i]:.4f}^+{q[0, i]:.4f}_{q[1, i]:.4f}"
                )

            print(
                f"Chi square: {self.eval_chi_square(result[1, :], npatch=npatch, apply_debias=apply_debias)}"
            )
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
        assert self.cov_tau is not None, (
            "Please load a covariance matrix for the tau statistics."
        )
        inv_cov = np.linalg.inv(self.cov_tau)
        if apply_debias:
            inv_cov = (npatch - tau_vec.shape[0] - 2) / (npatch - 1) * inv_cov
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

        fig, ax = self.tau_stat_handler.plot_tau_stats(
            [filename], [color], [catalog_id], plot_tau_m=False
        )

        assert self.rho_stat_handler.rho_stats is not None, (
            "Please load rho statistics data."
        )  # Check if data was loaded

        scales_diff_ratio = np.abs(
            (
                self.rho_stat_handler.rho_stats["theta"]
                - self.tau_stat_handler.tau_stats["theta"]
            )
            / self.tau_stat_handler.tau_stats["theta"]
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

        ylim = [[-4e-5, 4e-5], [-6e-6, 4e-6], [-7e-6, 1e-6]]
        for i in range(3):
            factor = (
                np.ones_like(self.tau_stat_handler.tau_stats["theta"])
                if i == 0
                else self.tau_stat_handler.tau_stats["theta"]
            )
            ax[0, i].plot(
                self.tau_stat_handler.tau_stats["theta"],
                taus[i] * factor,
                color="red",
                label="Model",
            )
            ax[0, i].legend(loc="upper right", fontsize="small")
            ax[0, i].set_ylim(ylim[i])
            ax[0, i].axhline(color="k", linestyle="dotted", linewidth=0.5)

        if savefig is not None:
            plt.savefig(self.data_directory + "/" + savefig)

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
            plt.errorbar(
                self.rho_stat_handler.rho_stats["theta"],
                xi_psf_sys,
                color=color,
                capsize=2,
                alpha=alpha,
            )
        else:
            plt.errorbar(
                self.rho_stat_handler.rho_stats["theta"],
                xi_psf_sys,
                color=color,
                capsize=2,
                alpha=alpha,
                label=r"$\xi^{\rm PSF}_{\rm sys, +}$ " + cat_id,
            )

        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel(r"$\theta$ [arcmin]")
        plt.ylabel(r"$\xi^{\rm PSF}_{\rm sys}$")

        if savefig:
            plt.savefig("xi_psf_sys.png")
            plt.close()

    def plot_xi_psf_sys_terms(self, cat_id, theta, out_path, yscale="log", show=False):
        ls = [
            "dotted",
            "dashed",
            "dashdot",
            (-1, (3, 5, 1, 5, 1, 5)),
            (0, (1, 10)),
            (0, (5, 5)),
        ]
        color = ["green", "blue", "red", "magenta", "cyan", "orange"]

        plt.figure(figsize=(15, 6))

        # self.plot_xi_psf_sys(theta, cat_id, "black")
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
            label=r"$\xi^{\rm PSF}_{\rm sys, +}$ " + cat_id,
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
                label=rf"{label_pre[term]} $\rho_{term}$",
            )
        xi_psf_sys_check = self.compute_xi_psf_sys(theta)

        plot_fct(ang_scales, xi_psf_sys_check, "bo", mfc="none")
        plot_fct(ang_scales, xi_sum, "bs", mfc="none")

        plt.xlabel(r"$\theta$ [arcmin]")
        plt.ylabel(r"$\xi^{\rm PSF}_{\rm sys}$")
        plt.legend(loc="best", fontsize="small")
        plt.ylim(ylim)
        plt.tight_layout()
        plt.savefig(out_path, bbox_inches="tight")
        if show:
            plt.show()
        plt.close()

    def _unpack_theta(self, theta):
        alpha, beta = theta[0], theta[1]
        idx = 2

        if self.use_eta:
            eta = theta[idx]
            idx += 1
        else:
            eta = 0.0

        if self.use_fourth_moment:
            alpha_4, beta_4 = theta[idx], theta[idx + 1]
        else:
            alpha_4, beta_4 = 0.0, 0.0

        return alpha, beta, eta, alpha_4, beta_4

    def compute_xi_psf_sys_term(self, theta_a, theta_b, term, p_or_m="p"):
        """
        Compute Xi Psf Sys Term.

        Compute one term of the systematic error on the cross-correlation given parameters (alpha, beta, eta)

        Parameters
        ----------
        theta_a : tuple
            Parameters (alpha, beta, (eta), (alpha_4, beta_4)) used to compute the systematic error from tomo_bin_a.

        theta_b : tuple
            Parameters (alpha, beta, (eta), (alpha_4, beta_4)) used to compute the systematic error from tomo_bin_b.

        term : int
            term (rho function) number, from 0 to 14

        p_or_m : str
            If "p", compute the systematic error on the xi+ correlation. If "m", compute the systematic error on the xi- correlation.

        Return
        ------
        np.array
            xi_psf_sys
        """
        alpha_a, beta_a, eta_a, alpha_4_a, beta_4_a = self._unpack_theta(theta_a)
        alpha_b, beta_b, eta_b, alpha_4_b, beta_4_b = self._unpack_theta(theta_b)
        if term == 0:
            prefactor = alpha_a * alpha_b
        elif term == 1:
            prefactor = beta_a * beta_b
        elif term == 3:
            prefactor = eta_a * eta_b
        elif term == 2:
            prefactor = alpha_a * beta_b + beta_a * alpha_b
        elif term == 5:
            prefactor = alpha_a * eta_b + eta_a * alpha_b
        elif term == 4:
            prefactor = beta_a * eta_b + eta_a * beta_b
        elif term == 6:
            prefactor = alpha_4_a * alpha_4_b
        elif term == 7:
            prefactor = alpha_a * alpha_4_b + alpha_4_a * alpha_b
        elif term == 8:
            prefactor = alpha_4_a * beta_b + beta_a * alpha_4_b
        elif term == 9:
            prefactor = beta_4_a + beta_4_b
        elif term == 10:
            prefactor = alpha_4_a * beta_4_b + beta_4_a * alpha_4_b
        elif term == 11:
            prefactor = beta_4_a * beta_b + beta_a * beta_4_b
        elif term == 12:
            prefactor = beta_4_a * alpha_b + alpha_a * beta_4_b
        elif term == 13:
            prefactor = alpha_4_a * eta_b + eta_a * alpha_4_b
        elif term == 14:
            prefactor = beta_4_a * eta_b + eta_a * beta_4_b
        else:
            raise ValueError(f"Invalid term {term}")
        if prefactor == 0:
            return np.zeros_like(self.rho_stat_handler.rho_stats["theta"])
        else:
            return prefactor * self.rho_stat_handler.rho_stats[f"rho_{term}_{p_or_m}"]

    def compute_xi_psf_sys(self, theta_a, theta_b=None, p_or_m="p"):
        """
        Compute Xi Psf Sys.

        Compute the systematic error on the cross-correlation given parameters (alpha, beta, eta)

        Parameters
        ----------
        theta_a : tuple
            Parameters (alpha, beta, (eta), (alpha_4, beta_4)) used to compute the systematic error from tomo bin a.

        theta_b : tuple
            Parameters (alpha, beta, (eta), (alpha_4, beta_4)) used to compute the systematic error from tomo bin b. If None, use tomo_bin_a only

        p_or_m : str
            If "p", compute the systematic error on the xi+ correlation. If "m", compute the systematic error on the xi- correlation.
        Return
        ------
        np.array
            xi_psf_sys
        """

        xi_psf_sys = np.zeros_like(self.rho_stat_handler.rho_stats["theta"])

        terms = [0, 1, 2]

        if self.use_eta:
            terms += [3, 4, 5]

        if self.use_fourth_moment:
            terms += [7, 8, 9, 10, 11, 12]

        if self.use_eta and self.use_fourth_moment:
            terms += [6, 13, 14]

        terms = sorted(terms)  # keep index 6 in the right place among the others

        if theta_b is None:
            theta_b = theta_a
        for term in terms:
            xi_psf_sys += self.compute_xi_psf_sys_term(
                theta_a, theta_b, term, p_or_m=p_or_m
            )

        return xi_psf_sys
