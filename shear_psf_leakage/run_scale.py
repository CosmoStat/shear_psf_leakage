"""RUN SCALE.

This module sets up a run of the scale-dependent leakage calculations.

:Author: Martin Kilbinger <martin.kilbinger@cea.fr>

"""

import os
from optparse import OptionParser

import numpy as np
from scipy.interpolate import CubicSpline
from lmfit import minimize, Parameters
import pandas as pd
from astropy.io import fits
from astropy import units
from uncertainties import ufloat, unumpy

from cs_util import logging
from cs_util import plots
from cs_util import calc
from cs_util import cat as cs_cat
from cs_util import cosmo as cs_cos
from cs_util import args as cs_args

from . import leakage
from . import correlation as corr


def get_theo_xi(theta, dndz_path):
    """Get Theo Xi.

    Return theoretical prediction of the shear 2PCF using a Planck
    best-fit cosmology.

    Parameters
    ----------
    theta : list
        angular scales, type is astropy.units.Quantity
    dndz_path : str
        input file path for redshift distribution

    Returns
    -------
    numpy.ndarray
        xi_+
    numpy.ndarray
        xi_-

    """
    z, nz, _ = cs_cat.read_dndz(dndz_path)
    cosmo = cs_cos.get_cosmo_default()
    xi_p, xi_m = cs_cos.xipm_theo(theta, cosmo, z, nz)
    

    return xi_p, xi_m

# MKDEBUG TODO: make class function
def save_alpha(theta, alpha_leak, sig_alpha_leak, sh, output_dir):
    """Save Alpha.

    Save scale-dependent alpha

    Parameters
    ----------
    theta : list
        angular scales
    alpha_leak : list
        leakage alpha(theta)
    sig_alpha_leak : list
        standard deviation of alpha(theta)
    sh : str
        shape measurement method, e.g. 'ngmix'
    output_dir : str
        output directory

    """
    cols = [theta, alpha_leak, sig_alpha_leak]
    names = ["# theta[arcmin]", "alpha", "sig_alpha"]
    fname = f"{output_dir}/alpha_leakage_{sh}.txt"
    write_ascii_table_file(cols, names, fname)

    
def save_xi_sys(
    theta,
    xi_sys_p,
    xi_sys_m,
    xi_sys_std_p,
    xi_sys_std_m,
    xi_p_theo,
    xi_m_theo,
    output_dir,
):
    """Save Xi Sys.

    Save 'xi_sys' cross-correlation function.

    Parameters
    ----------
    theta : list
        angular scales
    xi_sys_p : list
        xi+ component of cross-correlation function
    xi_sys_m : list
        xi- component of cross-correlation function
    xi_sys_std_p : list
        xi+ component of cross-correlation standard deviation
    xi_sys_std_m : list
        xi- component of cross-correlation standard deviation
    xi_p_theo : list
        xi+ component of theoretical shear-shear correlation
    xi_m_theo : list
        xi- component of theoretical shear-shear correlation
    output_dir : str
        output directory

    """
    cols = [
        theta,
        xi_sys_p,
        xi_sys_m,
        xi_sys_std_p,
        xi_sys_std_m,
        xi_p_theo,
        xi_m_theo,
    ]
    names = [
        "# theta[arcmin]",
        "xi_+_sys",
        "xi_-_sys",
        "sigma(xi_+_sys)",
        "sigma(xi_-_sys)",
        "xi_+_theo",
        "xi_-_theo",
    ]

    fname = f"{output_dir}/xi_sys.txt"
    cs_cat.write_ascii_table_file(cols, names, fname)


class LeakageScale:
    """Leakage Scale.

    Class to compute scale-dependent PSF leakage.

    """

    def __init__(self):
        # Set default parameters
        self.params_default()

    def set_params_from_command_line(self, args):
        """Set Params From Command Line.

        Only use when calling using python from command line.
        Does not work from ipython or jupyter.

        """
        # Read command line options
        options = cs_args.parse_options(
            self._params,
            self._short_options,
            self._types,
            self._help_strings,
        )

        # Update parameter values from options
        for key in vars(options):
            self._params[key] = getattr(options, key)

        # del options ?
        del options

        # Save calling command
        logging.log_command(args)

    def params_default(self):
        """Params Default.

        Set default parameter values.

        """
        self._params = {
            "input_path_shear": None,
            "e1_col": "e1",
            "e2_col": "e2",
            "w_col": None,
            "input_path_PSF": None,
            "hdu_psf": 1,
            "ra_star_col": "RA",
            "dec_star_col": "Dec",
            "e1_PSF_star_col": "E1_PSF_HSM",
            "e2_PSF_star_col": "E2_PSF_HSM",
            "dndz_path": None,
            "output_dir": ".",
            "close_pair_tolerance": None,
            "close_pair_mode": None,
            "cut": None,
            "theta_min_amin": 1,
            "theta_max_amin": 300,
            "n_theta": 20,
            "leakage_alpha_ylim": [-0.03, 0.1],
            "leakage_xi_sys_ylim": [-4e-5, 5e-5],
            "leakage_xi_sys_log_ylim": [2e-13, 5e-5],
        }

        self._short_options = {
            "input_path_shear": "-i",
            "input_path_PSF": "-I",
            "output_dir": "-o",
            "shapes": "-s",
            "close_pair_tolerance": "-t",
            "close_pair_mode": "-m",
        }

        self._types = {
            "hdu_psf": "int",
            "theta_min_amin": "float",
            "theta_max_amin": "float",
            "n_theta": "int",
        }

        self._help_strings = {
            "input_path_shear": "input path of the shear catalogue",
            "e1_col": "e1 column name in galaxy catalogue, default={}",
            "e2_col": "e2 column name in galaxy catalogue, default={}",
            "w_col": "weight column name in galaxy catalogue, default={}",
            "input_path_PSF": "input path of the PSF catalogue",
            "hdu_PSF": "HDU number of PSF catalogue, default={}",
            "ra_star_col": (
                "right ascension column name in star catalogue, default={}"
            ),
            "dec_star_col": (
                "declination column name in star catalogue, default={}"
            ),
            "e1_PSF_star_col": (
                "e1 PSF column name in star catalogue, default={}"
            ),
            "e2_PSF_star_col": (
                "e2 PSF column name in star catalogue, default={}"
            ),
            "dndz_path": (
                "path to galaxy redshift distribution file, for xi_sys ratio"
            ),
            "output_dir": "output_directory, default={}",
            "close_pair_tolerance": (
                "tolerance angle for close objects in star catalogue,"
                + " default={}"
            ),
            "close_pair_mode": (
                "mode for close objects in star catalogue, allowed are"
                + f" 'remove', 'average'"
            ),
            "cut": (
                "list of criteria (white-space separated, do not use '_')"
                + f" to cut data, e.g. 'w>0_mask!=0'"
            ),
            "theta_min_amin": "mininum angular scale [arcmin], default={}",
            "theta_max_amin": "maximum angular scale [arcmin], default={}",
            "n_theta": "number of angular scales on input, default={}",
        }

    def check_params(self):
        """Check Params.

        Check whether parameter values are valid.

        Raises
        ------
        ValueError
            if a parameter value is not valid

        """
        if not self._params["input_path_shear"]:
            raise ValueError("No input shear catalogue given")
        if not self._params["input_path_PSF"]:
            raise ValueError("No input star/PSF catalogue given")
        if not self._params["dndz_path"]:
            raise ValueError("No input n(z) file given")

        if "verbose" not in self._params:
            self._params["verbose"] = False

    def read_data(self):
        """Read Data.

        Read input galaxy and PSF catalogues.

        """
        # Read input shear
        dat_shear = self.read_shear_cat()

        # Apply cuts to galaxy catalogue if required
        dat_shear = leakage.cut_data(
            dat_shear, self._params["cut"], self._params["verbose"]
        )

        # Read star catalogue
        dat_PSF = leakage.open_fits_or_npy(
            self._params["input_path_PSF"],
            hdu_no=self._params["hdu_psf"],
        )

        # Deal with close objects in PSF catalogue (= stars on same position
        # from different exposures)
        dat_PSF = self.handle_close_objects(dat_PSF)

        # Set instance variables
        self.dat_shear = dat_shear
        self.dat_PSF = dat_PSF

    def prepare_output(self):
        """Prepare Output.

        Prepare output directory and stats file.

        """
        if not os.path.exists(self._params["output_dir"]):
            os.mkdir(self._params["output_dir"])
        self._stats_file = leakage.open_stats_file(
            self._params["output_dir"], "stats_file_leakage.txt"
        )

        for key in self._params:
            leakage.print_stats(
                f"{key}: {self._params[key]}", self._stats_file, False
            )

    def run(self):
        """Run.

        Main processing of scale-dependent leakage.

        """
        # Check parameter validity
        self.check_params()

        # Prepare output
        self.prepare_output()

        # Read input data
        self.read_data()

        # compute auto- and cross-correlation functions including alpha
        self.compute_corr_gp_pp_alpha()

        # alpha leakage
        self.do_alpha()

        # compute auto- and cross-correlation functions including alpha
        self.compute_corr_gp_pp_alpha_matrix()

        # alpha matrix leakage
        self.do_alpha_matrix()

        # xi_sys function
        self.do_xi_sys()

    def read_shear_cat(self):
        """Read Shear Cat.

        Read shear catalogue.

        """
        in_path = self._params["input_path_shear"]
        _, file_extension = os.path.splitext(in_path)
        if file_extension == ".parquet":
            df = pd.read_parquet(in_path, engine="pyarrow")
            sep_array = df["Separation"].to_numpy()
            idx = np.argwhere(np.isfinite(sep_array))
            dat_shear = {}
            for col in df:
                dat_shear[col] = df[col].to_numpy()[idx].flatten()
        else:
            hdu_list = fits.open(in_path)
            dat_shear = hdu_list[1].data
        n_shear = len(dat_shear)
        leakage.print_stats(
            f"{n_shear} galaxies found in shear catalogue",
            self._stats_file,
            verbose=self._params["verbose"],
        )

        return dat_shear

    def handle_close_objects(self, dat_PSF):
        """Handle Close Objects.

        Deal with close objects in PSF catalogue.

        Parameters
        ----------
        dat_PSF : FITS.record
            input PSF data

        Returns
        -------
        FITS.record
            processed PSF data

        """
        if not self._params["close_pair_tolerance"]:
            return dat_PSF

        n_star = len(dat_PSF)

        tolerance_angle = coords.Angle(self._params["close_pair_tolerance"])

        leakage.print_stats(
            f"close object distance = {tolerance_angle}",
            self._stats_file,
            verbose=self._params["verbose"],
        )

        # Create SkyCoord object from star positions
        coordinates = coords.SkyCoord(
            ra=dat_PSF[self._params["ra_star_col"]],
            dec=dat_PSF[self._params["dec_star_col"]],
            unit="deg",
        )

        # Search PSF catalogue in itself around tolerance angle
        indices1, indices2, d2d, d3d = coordinates.search_around_sky(
            coordinates, tolerance_angle
        )

        # Count multiplicity of indices = number of matches of search
        count = np.bincount(indices1)
        dat_PSF_proc = {}

        # Copy unique objects (multiplicity of unity)
        for col in dat_PSF.dtype.names:
            dat_PSF_proc[col] = dat_PSF[col][count == 1]
        n_non_close = len(dat_PSF_proc[self._params["ra_star_col"]])
        leakage.print_stats(
            f"found {n_non_close}/{n_star} = {n_non_close / n_star:.1%} "
            + "non-close objects",
            self._stats_file,
            verbose=self._params["verbose"],
        )

        # Deal with repeated objects (multiplicity > 1)
        multiples = count != 1
        if not multiples.any():
            # No multiples found -> no action
            leakage.print_stats(
                "no close objects found",
                self._stats_file,
                verbose=self._params["verbose"],
            )

        else:
            # Get index list of multiple objects
            idx_mult = np.where(multiples)[0]
            if self._params["mode"] == "average":
                # Initialise additional data vector
                dat_PSF_mult = {}
                for col in dat_PSF.dtype.names:
                    dat_PSF_mult[col] = []

                done = np.array([])
                n_avg_rem = 0

                # Loop over repeated indices
                for idx in idx_mult:
                    # If already used: ignore this index
                    if idx in done:
                        continue

                    # Get indices in data index list corresponding to
                    # this multiple index
                    w = np.where(indices1 == idx)[0]

                    # Get indices in data
                    ww = indices2[w]

                    # Append mean to additional data vector
                    for col in dat_PSF.dtype.names:
                        mean = np.mean(dat_PSF[col][ww])
                        dat_PSF_mult[col].append(mean)

                    # Register indixes to avoid repetition
                    done = np.append(done, ww)
                    n_avg_rem += len(ww) - 1

                n_avg = len(dat_PSF_mult[ra_star_col])
                leakage.print_stats(
                    f"adding {n_avg}/{n_star} = {n_avg / n_star:.1%} "
                    + "averaged objects",
                    self._stats_file,
                    verbose=self._params["verbose"],
                )

                for col in dat_PSF.dtype.names:
                    dat_PSF_proc[col] = np.append(
                        dat_PSF_proc[col], dat_PSF_mult[col]
                    )
            elif mode == "remove":
                n_rem = len(idx_mult)
                leakage.print_stats(
                    f"removing {n_rem}/{n_star} = {n_rem / n_star:.1%} "
                    + "close objects",
                    self._stats_file,
                    verbose=self._params["verbose"],
                )

        # Test
        coordinates_proc = coords.SkyCoord(
            ra=dat_PSF_proc[self._params["ra_star_col"]],
            dec=dat_PSF_proc[self._params["dec_star_col"]],
            unit="deg",
        )
        idx, d2d, d3d = coords.match_coordinates_sky(
            coordinates_proc, coordinates_proc, nthneighbor=2
        )
        non_close = (d2d > tolerance_angle).all()
        leakage.print_stats(
            f"Check: all remaining distances > {tolerance_angle}? {non_close}",
            self._stats_file,
            verbose=self._params["verbose"],
        )
        if mode == "average":
            leakage.print_stats(
                f"Check: n_non_close + n_avg + n_avg_rem = n_star? "
                + f"{n_non_close} + {n_avg} + {n_avg_rem} = "
                + f"{n_non_close + n_avg + n_avg_rem} ({n_star})",
                self._stats_file,
                verbose=self._params["verbose"],
            )
        elif mode == "remove":
            leakage.print_stats(
                f"Check: n_non_close + n_rem = n_star? {n_non_close} "
                + f"+ {n_rem} = {n_non_close + n_rem} ({n_star})",
                self._stats_file,
                verbose=self._params["verbose"],
            )

        n_in = len(dat_PSF[self._params["ra_star_col"]])
        n_out = len(dat_PSF_proc[self._params["dec_star_col"]])

        if n_in == n_out:
            leakage.print_stats(
                f"keeping all {n_out} stars",
                self._stats_file,
                verbose=self._params["verbose"],
            )
        else:
            leakage.print_stats(
                f"keeping {n_out}/{n_in} = {n_out/n_in:.1%} stars",
                self._stats_file,
                verbose=self._params["verbose"],
            )

        return dat_PSF_proc

    def get_theta(self):
        """Get Theta.

        Return angular scales.

        """
        return self.r_corr_gp.meanr

    def get_cat_fields(self, cat_type):
        """Get Cat Fields.

        Get catalogue fields for correlation.

        Parameters
        ----------
        cat_type : str
            catalogue type, allowed are "gal" and "star"

        Returns
        -------
        list
            ra
        list
            dec
        list
            e1
        list
            e2
        list
            weights; `None` if cat_type is "star"

        """
        if cat_type == "gal":
            ra = self.dat_shear["RA"]
            dec = self.dat_shear["Dec"]
            e1 = self.dat_shear[self._params["e1_col"]]
            e2 = self.dat_shear[self._params["e2_col"]]
            if self._params["w_col"] is not None:
                weights = self.dat_shear[self._params["w_col"]]
            else:
                weights = np.ones_like(ra)
        elif cat_type == "star":
            ra = self.dat_PSF[self._params["ra_star_col"]]
            dec = self.dat_PSF[self._params["dec_star_col"]]
            e1 = self.dat_PSF[self._params["e1_PSF_star_col"]]
            e2 = self.dat_PSF[self._params["e2_PSF_star_col"]]
            weights = None

        return ra, dec, e1, e2, weights

    def compute_corr_gp_pp_alpha(self, output_base_path=None):
        """Compute Corr GP PP Alpha.

        Compute and plot scale-dependent PSF leakage functions.

        Parameters
        ----------
        out_path : str, optional
                output file path; default is ``None`` (no file written)

        """
        ra, dec, e1_gal, e2_gal, weights = self.get_cat_fields("gal")
        ra_star, dec_star, e1_star, e2_star, _ = self.get_cat_fields("star")

        # Correlation functions
        r_corr_gp, r_corr_pp = corr.correlation_ab_bb(
            ra,
            dec,
            e1_gal,
            e2_gal,
            weights,
            ra_star,
            dec_star,
            e1_star,
            e2_star,
            theta_min_amin=self._params["theta_min_amin"],
            theta_max_amin=self._params["theta_max_amin"],
            n_theta=self._params["n_theta"],
            output_base_path=output_base_path,
        )

        # Check consistency of angular scales
        corr.check_consistency_scales(r_corr_gp, r_corr_pp)

        # Set instance variables
        self.r_corr_gp = r_corr_gp
        self.r_corr_pp = r_corr_pp

    def compute_corr_gp_pp_alpha_matrix(self):
        """Compute Corr GP PP Alpha Matrix.

        Compute and plot scale-dependent PSF leakage matrix.

        """
        ra, dec, e1_gal, e2_gal, weights = self.get_cat_fields("gal")
        ra_star, dec_star, e1_star, e2_star, _ = self.get_cat_fields("star")

        # Correlation functions
        r_corr_gp_m, r_corr_pp_m = corr.correlation_ab_bb_matrix(
            ra,
            dec,
            e1_gal,
            e2_gal,
            weights,
            ra_star,
            dec_star,
            e1_star,
            e2_star,
            theta_min_amin=self._params["theta_min_amin"],
            theta_max_amin=self._params["theta_max_amin"],
            n_theta=self._params["n_theta"],
        )

        # Check consistency of angular scales
        for idx in (0, 1):
            for jdx in (0, 1):
                corr.check_consistency_scales(
                    r_corr_gp_m[idx][jdx],
                    r_corr_pp_m[idx][jdx],
                )
                if idx != 0 and jdx != 0:
                    corr.check_consistency_scales(
                        r_corr_gp_m[0][0],
                        r_corr_gp_m[idx][jdx],
                    )
                    corr.check_consistency_scales(
                        r_corr_pp_m[0][0],
                        r_corr_pp_m[idx][jdx],
                    )

        if hasattr(self, "r_corr_gp"):
            corr.check_consistency_scales(self.r_corr_gp, r_corr_gp_m[0][0])

        self.r_corr_gp_m = r_corr_gp_m
        self.r_corr_pp_m = r_corr_pp_m

    def compute_alpha_mean(self):
        """Compute Alpha Mean.

        Compute weighted mean of the leakage function alpha.

        """
        self.alpha_leak_mean, self.alpha_leak_std = (
            calc.weighted_avg_and_std(
                self.alpha_leak,
                1/self.sig_alpha_leak**2
            ) 
        )
        #calc.transform_nan(
        #    np.average(self.alpha_leak, weights=1/self.sig_alpha_leak**2)
        #)
        leakage.print_stats(
            f"Weighted average alpha" + f" = {self.alpha_leak_mean:.3g}",
            self._stats_file,
            verbose=self._params["verbose"],
        )

    def alpha_affine_fit(self):
        """Alpha Affine Fit.

        Fit alpha(theta) with an affine model.

        """

        params = Parameters()
        params.add("m", value=0.01)
        params.add("c", value=0.01)
        res = minimize(
            leakage.loss_bias_lin_1d,
            params,
            args=(self.r_corr_gp.meanr, self.alpha_leak, self.alpha_leak_std)
        )

        # Save best-fit parameters
        self.alpha_affine_best_fit = res.params
        self.alpha_leak_zero = leakage.func_bias_lin_1d(
            self.alpha_affine_best_fit,
            0,
        )

    def compute_xi_sys(self):
        """Compute Xi Sys.

        Compute galaxy - PSF systematics correlation function.

        """
        C_sys_p = self.r_corr_gp.xip**2 / self.r_corr_pp.xip
        C_sys_m = self.r_corr_gp.xim**2 / self.r_corr_pp.xim

        term_gp = (2 / self.r_corr_gp.xip) ** 2 * self.r_corr_gp.varxip
        term_pp = (1 / self.r_corr_pp.xip) ** 2 * self.r_corr_pp.varxip
        C_sys_std_p = np.abs(C_sys_p) * np.sqrt(term_gp + term_pp)

        term_gp = (2 / self.r_corr_gp.xim) ** 2 * self.r_corr_gp.varxim
        term_pp = (1 / self.r_corr_pp.xim) ** 2 * self.r_corr_pp.varxim
        C_sys_std_m = np.abs(C_sys_m) * np.sqrt(term_gp + term_pp)

        self.C_sys_p = C_sys_p
        self.C_sys_m = C_sys_m
        self.C_sys_std_p = C_sys_std_p
        self.C_sys_std_m = C_sys_std_m

    def plot_alpha_leakage(self, close_fig=True, xlinlog="log"):
        """Plot Alpha Leakage.

        Plot scale-dependent leakage function alpha(theta)

        Parameters
        -----------
        close_fig : bool, optional
            closes figure if ``True``; set this parameter to ``False`` for
            notebook display; default is ``True``
        xlinlog : str, optional
            if "lin" ("log"; default), creat plot linear (logarithmic) in x

        """
        # Set some x-values and limits for plot
        if xlinlog == "log":
            x0 = self._params["theta_min_amin"]
            x_affine = np.geomspace(x0, self._params["theta_max_amin"])
            xlim = [x0, self._params["theta_max_amin"]]
            xlog = True
        else:
            x0 = -10
            x_affine = np.linspace(0, self._params["theta_max_amin"])
            xlim = [x0 * 2, self._params["theta_max_amin"]]
            xlog = False

        # alpha(theta) data points
        theta = [self.r_corr_gp.meanr]
        alpha_theta = [self.alpha_leak]
        yerr = [self.sig_alpha_leak]

        # mean alpha
        theta.append([x0])
        alpha_theta.append([self.alpha_leak_mean])
        yerr.append([self.alpha_leak_std])

        labels = [r"$\alpha(\theta)$", r"$\bar\alpha$"]
        linewidths = [2, 2]

        # affine model best fit alpha(0)
        theta.append([x0])
        alpha_theta.append([self.alpha_affine_best_fit["c"].value])
        yerr.append([self.alpha_affine_best_fit["c"].stderr])
        labels.append(r"$\alpha(0)$")
        linewidths.append(1)

        # affine model best fit alpha(theta)
        theta.append(x_affine)
        alpha_theta.append(
            leakage.func_bias_lin_1d(self.alpha_affine_best_fit, x_affine)
        )
        # MKDEBUG TODO: error band
        yerr.append(np.zeros_like(x_affine) * np.nan)
        labels.append(r"$\alpha(\theta)$ affine fit")
        linewidths.append(1)

        xlabel = r"$\theta$ [arcmin]"
        ylabel = r"$\alpha(\theta)$"
        title = ""
        out_path = (
            f"{self._params['output_dir']}" + f"/alpha_leakage_{xlinlog}.png"
        )
        ylim = self._params["leakage_alpha_ylim"]

        plots.plot_data_1d(
            theta,
            alpha_theta,
            yerr,
            title,
            xlabel,
            ylabel,
            out_path,
            xlog=xlog,
            xlim=xlim,
            ylim=ylim,
            labels=labels,
            linewidths=linewidths,
            close_fig=close_fig,
            shift_x=False,
        )

    def plot_alpha_leakage_matrix(self):
        """Plot Alpha Leakage Matrix.

        Plot scale-dependent leakage matrix alpha.

        """
        theta = self.get_theta()
        theta_arr = []
        alpha_arr = []
        yerr_arr = []
        labels = []
        ftheta = 1.05
        n = 4
        count = 0
        for idx in (0, 1):
            for jdx in (0, 1):
                theta_arr.append(theta * ftheta ** (count - n))
                alpha = self.get_alpha_ufloat(idx, jdx)
                alpha_arr.append(unumpy.nominal_values(alpha))
                yerr_arr.append(unumpy.std_devs(alpha))
                labels.append(rf"$\alpha_{{{idx+1}{jdx+1}}}$")
        xlabel = r"$\theta$ [arcmin]"
        ylabel = r"$\alpha_{ij}(\theta)$"
        title = ""
        out_path = f"{self._params['output_dir']}/alpha_leakage_matrix.png"
        xlim = [self._params["theta_min_amin"], self._params["theta_max_amin"]]
        ylim = self._params["leakage_alpha_ylim"]
        plots.plot_data_1d(
            theta_arr,
            alpha_arr,
            yerr_arr,
            title,
            xlabel,
            ylabel,
            out_path,
            xlog=True,
            xlim=xlim,
            ylim=ylim,
            labels=labels,
        )

    def plot_xi_sys(self, close_fig=True):
        """Plot Xi Sys.

        Plot galaxy - PSF systematics correlation function.

        Parameters
        -----------
        close_fig : bool, optional
            closes figure if ``True``; set this parameter to ``False`` for
            notebook display; default is ``True``

        """
        labels = ["$\\xi^{\\rm sys}_+$", "$\\xi^{\\rm sys}_-$"]

        title = "Cross-correlation leakage"
        xlabel = "$\\theta$ [arcmin]"
        ylabel = "Correlation function"

        theta = [self.r_corr_gp.meanr] * 2
        xi = [self.C_sys_p, self.C_sys_m]
        yerr = [self.C_sys_std_p, self.C_sys_std_m]

        comp_arr = [0, 1]
        symb_arr = ["+", "-"]
        for comp, symb in zip(comp_arr, symb_arr):
            mean = np.mean(np.abs(xi[comp]))
            msg = f"<|xi_sys_{symb}|> = {mean}"
            leakage.print_stats(
                msg, self._stats_file, verbose=self._params["verbose"]
            )

        ylim = self._params["leakage_xi_sys_ylim"]
        out_path = f"{self._params['output_dir']}/xi_sys.pdf"
        plots.plot_data_1d(
            theta,
            xi,
            yerr,
            title,
            xlabel,
            ylabel,
            out_path,
            xlog=True,
            ylim=ylim,
            labels=labels,
            close_fig=close_fig,
        )

        ylim = self._params["leakage_xi_sys_log_ylim"]
        out_path = f"{self._params['output_dir']}/xi_sys_log.pdf"
        plots.plot_data_1d(
            theta,
            xi,
            yerr,
            title,
            xlabel,
            ylabel,
            out_path,
            xlog=True,
            ylog=True,
            ylim=ylim,
            labels=labels,
            close_fig=close_fig,
        )

    def plot_xi_sys_ratio(self, xi_p_theo, xi_m_theo, close_fig=True):
        """Plot Xi Sys Ratio.

        Plot xi_sys relative to theoretical model of cosmological
        xi_pm.

        Parameters
        ----------
        xi_p_theo : list
            theoretical model of xi+
        xi_m_theo : list
            theoretical model of xi-
        close_fig : bool, optional
            closes figure if ``True``; set this parameter to ``False`` for
            notebook display; default is ``True``

        """
        labels = [
            "$\\xi^{\\rm sys}_+ / \\xi_+$",
            "$\\xi^{\\rm sys}_- / \\xi_-$",
        ]

        title = "Cross-correlation leakage ratio"
        xlabel = "$\\theta$ [arcmin]"
        ylabel = "Correlation function ratio"

        theta = [self.r_corr_gp.meanr] * 2
        xi = [self.C_sys_p / xi_p_theo, self.C_sys_m / xi_m_theo]
        yerr = [
            self.C_sys_std_p / np.abs(xi_p_theo),
            self.C_sys_std_m / np.abs(xi_m_theo),
        ]

        comp_arr = [0, 1]
        symb_arr = ["+", "-"]
        for comp, symb in zip(comp_arr, symb_arr):
            mean = np.mean(np.abs(xi[comp]))
            msg = f"<|xi_sys_{symb}| / xi_{symb}> = {mean}"
            leakage.print_stats(
                msg, self._stats_file, verbose=self._params["verbose"]
            )

        out_path = f"{self._params['output_dir']}" + f"/xi_sys_ratio.pdf"

        ylim = [0, 0.5]

        plots.plot_data_1d(
            theta,
            xi,
            yerr,
            title,
            xlabel,
            ylabel,
            out_path,
            xlog=True,
            ylim=ylim,
            labels=labels,
            close_fig=close_fig,
        )

    def do_alpha(self, fast=False):
        """Do Alpha.

        Compute, plot, and save alpha leakage function.

        Parameters
        ----------
        fast: bool, optional
            omits (time-consuming) calculation of mean ellipticity and neglects
            those small terms if True; default is ``False``

        """
        # Get input catalogues for averages
        _, _, e1_gal, e2_gal, weights = self.get_cat_fields("gal")
        _, _, e1_star, e2_star, _ = self.get_cat_fields("star")

        # Compute alpha leakage function
        self.alpha_leak, self.sig_alpha_leak = corr.alpha(
            self.r_corr_gp,
            self.r_corr_pp,
            e1_gal,
            e2_gal,
            weights,
            e1_star,
            e2_star,
            fast=fast,
        )
        self.compute_alpha_mean()
        self.alpha_affine_fit()

        # Plot
        for xlinlog in ["lin", "log"]:
            self.plot_alpha_leakage(xlinlog=xlinlog)

        # Write to disk
        self.save_alpha()

    def alpha_matrix(self):
        """Alpha Matrix.

        Compute alpha leakage matrix.

        """
        _, _, e1_gal, e2_gal, weights = self.get_cat_fields("gal")
        _, _, e1_star, e2_star, _ = self.get_cat_fields("star")

        # <e^g>
        e_g = np.matrix(
            [
                np.average(e1_gal, weights=weights),
                np.average(e2_gal, weights=weights),
            ]
        )

        # <e^p>
        e_p = np.matrix(
            [
                np.mean(e1_star),
                np.mean(e2_star),
            ]
        )

        n_theta = self._params["n_theta"]

        # Set correlation function matrices
        self.xi_gp_m = np.zeros((2, 2, n_theta))
        self.xi_pp_m = np.zeros((2, 2, n_theta))
        for idx in (0, 1):
            for jdx in (0, 1):
                self.xi_gp_m[idx][jdx] = self.r_corr_gp_m[idx][jdx].xip
                self.xi_pp_m[idx][jdx] = self.r_corr_pp_m[idx][jdx].xip

        # Set centered correlation function matrices
        self.Xi_gp_m = np.zeros((2, 2, n_theta))
        self.Xi_pp_m = np.zeros((2, 2, n_theta))
        for ndx in range(n_theta):
            self.Xi_gp_m[:, :, ndx] = self.xi_gp_m[:, :, ndx] - np.dot(
                e_g.transpose(), e_p
            )
            self.Xi_pp_m[:, :, ndx] = self.xi_pp_m[:, :, ndx] - np.dot(
                e_p.transpose(), e_p
            )

        # Standard deviations
        self.xi_std_gp_m = np.zeros((2, 2, n_theta))
        self.xi_std_pp_m = np.zeros((2, 2, n_theta))
        for idx in (0, 1):
            for jdx in (0, 1):
                self.xi_std_gp_m[idx][jdx] = np.sqrt(
                    self.r_corr_gp_m[idx][jdx].varxip
                )
                self.xi_std_pp_m[idx][jdx] = np.sqrt(
                    self.r_corr_pp_m[idx][jdx].varxip
                )

        # TODO: include <e><e> in error computation

        self.Xi_gp_ufloat = []
        self.Xi_pp_ufloat = []
        values = np.zeros((2, 2), dtype=float)
        stds = np.zeros((2, 2), dtype=float)
        for ndx in range(n_theta):

            # Set Xi_gp
            for idx in (0, 1):
                for jdx in (0, 1):
                    values[idx, jdx] = self.Xi_gp_m[idx, jdx, ndx]
                    stds[idx, jdx] = self.xi_std_gp_m[idx][jdx][ndx]

            self.Xi_gp_ufloat.append(unumpy.umatrix(values, stds))

            # Set Xi_pp
            for idx in (0, 1):
                for jdx in (0, 1):
                    values[idx, jdx] = self.Xi_pp_m[idx, jdx, ndx]
                    stds[idx, jdx] = self.xi_std_pp_m[idx][jdx][ndx]

            self.Xi_pp_ufloat.append(unumpy.umatrix(values, stds))

        # Compute (Xi_pp)^{-1} and alpha
        self.Xi_pp_inv_ufloat = []
        self.alpha_leak_ufloat = []
        for ndx in range(n_theta):
            self.Xi_pp_inv_ufloat.append(self.Xi_pp_ufloat[ndx].I)
            self.alpha_leak_ufloat.append(
                np.dot(self.Xi_gp_ufloat[ndx], self.Xi_pp_inv_ufloat[ndx])
            )

    def get_alpha_ufloat(self, idx, jdx):
        """Get Alpha Ufloat.

        Return alpha leakage matrix element as array over scales.

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
            mat.append(self.alpha_leak_ufloat[ndx][idx, jdx])

        return np.array(mat)

    def do_alpha_matrix(self):
        """Do Alpha Matrix.

        Compute, plot, and save alpha leakage matrix.

        """
        # Compute
        self.alpha_matrix()

        # Plot
        self.plot_alpha_leakage_matrix()

        # Write to disk
        self.save_alpha_matrix()

    def do_xi_sys(self):
        """Do Xi Sys.

        Compute, plot, and save xi_sys function.

        """
        # Compute xi_sys
        self.compute_xi_sys()

        # Compute theoretical model for the 2PCF

        # Treecorr output scales are in arc minutes
        theta = self.r_corr_gp.meanr * units.arcmin
        xi_p_theo, xi_m_theo = get_theo_xi(theta, self._params["dndz_path"])

        # Plot
        self.plot_xi_sys()
        self.plot_xi_sys_ratio(xi_p_theo, xi_m_theo)

        # Write to disk
        save_xi_sys(
            self.r_corr_gp.meanr,
            self.C_sys_p,
            self.C_sys_m,
            self.C_sys_std_p,
            self.C_sys_std_m,
            xi_p_theo,
            xi_m_theo,
            self._params["output_dir"],
        )

    def save_r_corr_ab(self):
        """Save R Corr AB.

        Save correlation function.

        """

    def save_alpha(self):
        """Save Alpha.

        Save scale-dependent alpha.

        """
        cols = [self.r_corr_gp.meanr, self.alpha_leak, self.sig_alpha_leak]
        names = ["# theta[arcmin]", "alpha", "sig_alpha"]
        fname = f"{self._params['output_dir']}/alpha_leakage.txt"
        cs_cat.write_ascii_table_file(cols, names, fname)

    def save_alpha_matrix(self):
        """Save Alpha Matrix.

        Save scale-dependent alpha matrix.

        """
        cols = [self.get_theta()]
        names = ["# theta[arcmin]"]
        for idx in (0, 1):
            for jdx in (0, 1):
                alpha = self.get_alpha_ufloat(idx, jdx)
                cols.append(unumpy.nominal_values(alpha))
                names.append(rf"alpha_{{{idx+1}{jdx+1}}}")
                cols.append(unumpy.std_devs(alpha))
                names.append(rf"sigma_alpha_{{{idx+1}{jdx+1}}}")
        fname = f"{self._params['output_dir']}/alpha_leakage_matrix.txt"
        cs_cat.write_ascii_table_file(cols, names, fname)


def run_leakage_scale(*args):
    """Run Leakage Scale.

    Run scale-dependent PSF leakage as python script from command line.

    """
    # Create object for scale-dependent leakage calculations
    obj = LeakageScale()

    obj.set_params_from_command_line(args)

    obj.run()
