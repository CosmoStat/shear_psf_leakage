import os
import numpy as np

from astropy.io import fits
from lmfit import Parameters

from optparse import OptionParser

from cs_util import logging

from . import leakage


# MKDEBUG TODO: to cs_util. Also check shapepipe.
def my_string_split(string, num=-1, verbose=False, stop=False, sep=None):
    """My String Split.

    Split a *string* into a list of strings. Choose as separator
    the first in the list [space, underscore] that occurs in the string.
    (Thus, if both occur, use space.)

    Parameters
    ----------
    string : str
        Input string
    num : int
        Required length of output list of strings, -1 if no requirement.
    verbose : bool
        Verbose output
    stop : bool
        Stop programs with error if True, return None and continues otherwise
    sep : bool
        Separator, try ' ', '_', and '.' if None (default)

    Raises
    ------
    CfisError
        If number of elements in string and num are different, for stop=True
    ValueError
        If no separator found in string

    Returns
    -------
    list
        List of string on success, and None if failed

    """
    if string is None:
        return None

    if sep is None:
        has_space = string.find(" ")
        has_underscore = string.find("_")
        has_dot = string.find(".")

        if has_space != -1:
            my_sep = " "
        elif has_underscore != -1:
            my_sep = "_"
        elif has_dot != -1:
            my_sep = "."
        else:
            # no separator found, does string consist of only one element?
            if num == -1 or num == 1:
                my_sep = None
            else:
                raise Valueerror(
                    "No separator (' ', '_', or '.') found in string"
                    + f" '{string}', cannot split"
                )
    else:
        if not string.find(sep):
            raise ValueError(
                f"No separator '{sep}' found in string '{string}', " + "cannot split"
            )
        my_sep = sep

    res = string.split(my_sep)

    if num != -1 and num != len(res) and stop:
        raise CfisError(f"String '{len(res)}' has length {num}, required is {num}")

    return res


# MKDEBUG to cs_util
def parse_options(p_def, short_options, types, help_strings):
    """Parse Options.

    Parse command line options.

    Parameters
    ----------
    p_def : dict
        default parameter values
    help_strings : dict
        help strings for options

    Returns
    -------
    options: tuple
        Command line options
    """

    usage = "%prog [OPTIONS]"
    parser = OptionParser(usage=usage)

    for key in p_def:
        if key in help_strings:
            if key in short_options:
                short = short_options[key]
            else:
                short = ""

            if key in types:
                typ = types[key]
            else:
                typ = "string"

            if typ == "bool":
                parser.add_option(
                    f"{short}",
                    f"--{key}",
                    dest=key,
                    default=False,
                    action="store_true",
                    help=help_strings[key].format(p_def[key]),
                )
            else:
                parser.add_option(
                    f"{short}",
                    f"--{key}",
                    dest=key,
                    type=typ,
                    default=p_def[key],
                    help=help_strings[key].format(p_def[key]),
                )

    parser.add_option(
        "-v",
        "--verbose",
        dest="verbose",
        action="store_true",
        help=f"verbose output",
    )

    options, args = parser.parse_args()

    return options


class LeakageObject:
    """LeakageObject.

    Class to compute object-wise leakage with the PSF and other quantities.

    """

    def __init__(self):
        # Set default parameters
        self.params_default()

    def set_params_from_command_line(self, args):
        """Set Params From Command line.

        Only use when calling using python from command line.
        Does not work from ipython or jupyter.

        """
        # Read command line options
        options = parse_options(
            self._params,
            self._short_options,
            self._types,
            self._help_strings,
        )

        # Update parameter values from options
        for key in vars(options):
            self._params[key] = getattr(options, key)

        # del options ?

        # Save calling command
        logging.log_command(args)

    def params_default(self):
        """Params Default.

        Set default parameter values.

        """
        # MKDEBUG TODO: test -> action=store_true
        self._params = {
            "input_path_shear": None,
            "output_dir": ".",
            "e1_col": "e1_uncal",
            "e2_col": "e2_uncal",
            "e1_PSF_col": "e1_PSF",
            "e2_PSF_col": "e2_PSF",
            "size_PSF_col": "fwhm_PSF",
            "RA_col": "RA",
            "Dec_col": "Dec",
            "PSF_leakage": True,
            "obs_leakage": False,
            "cols": None,
            "cols_ratio": None,
            "test": False,
        }
        self._short_options = {
            "input_path_shear": "-i",
            "output_dir": "-o",
            "test": "-t",
        }
        self._types = {
            "test": "bool",
        }
        self._help_strings = {
            "input_path_shear": "input path of the extended shear catalogue",
            "output_dir": "output_dir, default={}",
            "e1_col": "e1 column name in galaxy catalogue, default={}",
            "e2_col": "e2 column name in galaxy catalogue, default={}",
            "e1_PSF_col": "PSF e1 column name in galaxy catalogue, default={}",
            "e2_PSF_col": "PSF e2 column name in galaxy catalogue, default={}",
            "size_PSF_col": "PSF size column name, default={}",
            "RA_col": "right ascension column name, default={}",
            "Dec_col": "declination column name, default={}",
            "PSF_leakage": "Fit spin-2 consistent PSF leakage relations",
            "obs_leakage": "Fit leakage relations with abitrary observables",
            "cols": "White-space separated list of column names for fit",
            "cols_ratio": "fit as function of ratio of two columns",
            "test": "Fit toy model and exit",
        }

    def check_params(self):
        """Check Params.

        Check whether parameter values are valid.

        Raises
        ------
        ValueError
            if a parameter value is not valid

        """
        if not self._params["input_path_shear"] and not self._params["test"]:
            raise ValueError(
                "Input path for shear catalogue (option '-i') "
                + "required unless in test mode (option '-t')"
            )

        if (
            not self._params["PSF_leakage"]
            and not self._params["obs_leakage"]
            and not self._params["test"]
        ):
            raise ValueError(
                "Required options are '-t' xor ('--PSF_leakage' or "
                + " '--obs_Leakage'"
            )

        if not self._params["obs_leakage"] and self._params["cols_ratio"]:
            raise ValueError(f"Option 'cols_ratio' only valid for obs_leakage")

        if self._params["e1_col"] == self._params["e2_col"]:
            raise ValueError(
                "Column names for e1 and e2 are identical, "
                + "this is surely a mistake"
            )
        if self._params["e1_PSF_col"] == self._params["e2_PSF_col"]:
            raise ValueError(
                "Column names for e1_PSF and e2_PSF are identical, "
                + "this is surely a mistake"
            )

    def update_params(self):
        """Update Params.

        Update parameters.

        """
        if self._params["cols"]:
            self._params["cols"] = my_string_split(
                self._params["cols"],
                verbose=self._params["verbose"],
                stop=True,
            )
        if self._params["cols_ratio"]:
            self._params["cols_ratio"] = my_string_split(
                self._params["cols_ratio"],
                num=2,
                verbose=self._params["verbose"],
                stop=True,
            )

    def prepare_output(self):
        """Prepare Output.

        Prepare output directory and stats file.

        """
        # Creation of the output directory
        if not os.path.exists(self._params["output_dir"]):
            os.mkdir(self._params["output_dir"])

        # Creation of the statistics file handler
        self._stats_file = leakage.open_stats_file(
            self._params["output_dir"],
            "stats_file_leakage.txt",
        )

    def read_data(self):
        """Read Data.

        Read input catalogue with galaxy and PSF information.

        """
        # Open Fits file of the input shear catalogue
        hdu_list = fits.open(self._params["input_path_shear"])
        self._dat = hdu_list[1].data
        hdu_list.close()

    def corr_any_quant(self, label_quant=None, ratio=None):
        """Corr_any_quant.

        Compute and plot object-by-object ellipticity and any quantities relations.
        Plot also a recap plot of all slopes of the best fits of the e_gal vs
        other quantities.

        Parameters
        ----------
        label_quant : str, optional
        ratio : bool, optional

        """
        # Set plotting options
        n_bin = 30
        colors = ["b", "r"]
        ylabel = r"$e_{1,2}^{\rm gal}$"
        mlabel = [r"\alpha_1", r"\alpha_2"]
        clabel = ["c_1", "c_2"]

        e, weights = self.get_ellipticity_weights()

        x_arr = []
        out_name_arr = []
        xlabel_arr = []

        if label_quant:
            xlabel_arr = label_quant
            for colname in xlabel_arr:
                x_arr.append(self._dat[colname])
                out_name_arr.append(colname + "_vs_e_gal")

        if ratio:
            x_arr.append(self._dat[ratio[0]] / self._dat[ratio[1]])
            xlabel_arr.append(f"{ratio[0]}/{ratio[1]}")
            out_name_arr.append(f"{ratio[0]}_div_{ratio[1]}_vs_e_gal")

        if self._params["verbose"]:
            print("Quadratic fit")
        out_path_arr = [
            f"{self._params['output_dir']}/{name}_quad" for name in out_name_arr
        ]
        name = "systematics_test_quad"
        out_path_arr.append(f"{self._params['output_dir']}/{name}")
        qlabel = ["q_1", "q_2"]
        leakage.quad_corr_n_quant(
            x_arr,
            e,
            xlabel_arr,
            ylabel,
            qlabel=qlabel,
            mlabel=mlabel,
            clabel=clabel,
            title="quadratic model",
            weights=weights,
            n_bin=n_bin,
            out_path_arr=out_path_arr,
            colors=colors,
            stats_file=self._stats_file,
            verbose=self._params["verbose"],
        )

        if self._params["verbose"]:
            print("Linear fit")
        out_path_arr = [
            f"{self._params['output_dir']}/{name}_lin" for name in out_name_arr
        ]
        name = "systematics_test_lin"
        out_path_arr.append(f"{self._params['output_dir']}/{name}")
        leakage.affine_corr_n(
            x_arr,
            e,
            xlabel_arr,
            ylabel,
            mlabel=mlabel,
            clabel=clabel,
            title="linear model",
            weights=weights,
            n_bin=n_bin,
            out_path_arr=out_path_arr,
            colors=colors,
            stats_file=self._stats_file,
            verbose=self._params["verbose"],
        )

    def test(self):
        """Test

        Test 2D spin-conserving object-by-object leakage relations.

        """
        # Set options for plotting

        # Number of bins in x, for plotting only
        n_bin = 20

        colors = ["b", "r"]
        ylabel_arr = ["$y_1$", "$y_2$"]

        xlabel_arr = [
            r"$x_1$",
            r"$x_2$",
        ]

        # Initialise random generator
        np.random.seed(seed=6121975)

        # Set 2D (x_1, x_2) values
        xm = 1.0
        size = 2000
        sig_x = 0.5
        x_arr = [
            np.random.uniform(-xm, xm, size=size),
            np.random.uniform(-xm, xm, size=size),
        ]

        # Ground truth parameter values
        pars_gt = {
            "q111": -0.9,
            "q222": 0.3,
            "q112": 1.8,
            "q122": -1.3,
            "q212": -2.0,
            "q211": 0.25,
            "a11": -0.4,
            "a22": 0.3,
            "a12": 0.3,
            "c1": 0.2,
            "c2": -0.3,
        }
        p_gt = Parameters()
        for par in pars_gt:
            p_gt.add(par, value=pars_gt[par])

        # Ground-truth 2D (y_1, y_2) data
        y1, y2 = leakage.func_bias_2d(p_gt, x_arr[0], x_arr[1], order="quad", mix=True)

        # Perturbation
        dy1 = np.random.normal(scale=sig_x, size=size)
        dy2 = np.random.normal(scale=sig_x, size=size)

        # Perform fits
        for order in ["lin", "quad"]:
            for mix in [False, True]:
                out_path = f"{self._params['output_dir']}/test_{order}_{mix}"
                leakage.corr_2d(
                    x_arr,
                    [y1 + dy1, y2 + dy2],
                    xlabel_arr=xlabel_arr,
                    ylabel_arr=ylabel_arr,
                    order=order,
                    mix=mix,
                    title=f"test {order} {mix}",
                    n_bin=n_bin,
                    out_path=out_path,
                    colors=colors,
                    y_ground_truth=[y1, y2],
                    par_ground_truth=p_gt,
                    stats_file=self._stats_file,
                    verbose=self._params["verbose"],
                )

        print("Ground truth:")
        for par in p_gt:
            print(par, p_gt[par].value)

    def get_ellipticity_weights(self):
        """Get Ellipticity Weights.

        Return galaxy ellipticity and weights from input data.

        """
        e1 = self._dat[self._params["e1_col"]]
        e2 = self._dat[self._params["e2_col"]]
        e = np.array([e1, e2])
        weights = self._dat["w"]

        return e, weights

    def PSF_leakage(self):
        """PSF Leakage.

        Compute and plot object-by-object PSF spin-consistent leakage relations.

        """
        # Set options for plotting
        n_bin = 30
        colors = ["b", "r"]
        ylabel_arr = [r"$e_1^{\rm gal}$", r"$e_2^{\rm gal}$"]

        xlabel_arr = [
            r"$e_{1}^{\rm PSF}$",
            r"$e_{2}^{\rm PSF}$",
            r"$\mathrm{FWHM}^{\rm PSF}$ [arcsec]",
        ]

        e, weights = self.get_ellipticity_weights()

        x_arr = [
            self._dat[self._params["e1_PSF_col"]],
            self._dat[self._params["e2_PSF_col"]],
            self._dat[self._params["size_PSF_col"]],
        ]
        out_name_arr = [
            "PSF_e1_vs_e_gal",
            "PSF_e2_vs_e_gal",
            "PSF_size_vs_e_gal",
        ]

        # Fit consistent spin-2 2D model
        mix = True
        for order in ["lin", "quad"]:
            out_path = (
                f"{self._params['output_dir']}"
                + "/PSF_e_vs_e_gal_order-{order}_mix-{mix}"
            )
            par_best_fit = leakage.corr_2d(
                x_arr[:2],
                e,
                weights=weights,
                xlabel_arr=xlabel_arr[:2],
                ylabel_arr=ylabel_arr,
                order=order,
                mix=mix,
                title=f"{order} {mix}",
                n_bin=n_bin,
                out_path=out_path,
                colors=colors,
                stats_file=self._stats_file,
                verbose=self._params["verbose"],
            )
            fp_best_fit = open(f"{out_path}.json", "w")
            par_best_fit.dump(fp_best_fit)

        # Fit separate 1D models, including size
        ylabel = r"$e_{1,2}^{\rm gal}$"
        mlabel = [r"\alpha_1", r"\alpha_2"]
        clabel = ["c_1", "c_2"]

        out_path_arr = [f"{self._params['output_dir']}/{name}" for name in out_name_arr]
        name = "systematics_test"
        out_path_arr.append(f"{self._params['output_dir']}/{name}")
        leakage.affine_corr_n(
            x_arr,
            e,
            xlabel_arr,
            ylabel,
            mlabel=mlabel,
            clabel=clabel,
            title="",
            weights=weights,
            n_bin=n_bin,
            out_path_arr=out_path_arr,
            colors=colors,
            stats_file=self._stats_file,
            verbose=self._params["verbose"],
        )

    def obs_leakage(self):
        """Obs Leakage.

        Compute and plot object-by-object "leakage" relations with arbitrary
        input quantities.
        Plot also a recap plot of all slopes of the best fits of the e_gal vs quantities

        """
        # Get quantities to fix
        if not self._params["cols"]:
            # Get user input
            print("Data columns names :")
            print(self._dat.dtype.names)
            change_header = input(
                "Enter list of columns (comma-separated, no whitespaces: "
            )
            label_quant = [str(col) for col in change_header.split(",")]
        else:
            # Use command line argument
            label_quant = self._params["cols"]

        # Remove duplicates
        label_quant = list(set(label_quant))

        print("columns selected:", label_quant, end="")
        if self._params["cols_ratio"]:
            print(
                " ", self._params["cols_ratio"][0], "/", self._params["cols_ratio"][1]
            )
        self.corr_any_quant(label_quant, ratio=self._params["cols_ratio"])

    def run(self):
        """Run.

        Main processing of scale-dependent leakage.

        """
        # To use same name as in notebook
        obj = self

        # Check parameter validity
        obj.check_params()

        # Update parameters (here: strings to list)
        obj.update_params()

        # Prepare output directory
        obj.prepare_output()

        if obj._params["test"]:
            # 2D spin-consistent test fit
            obj.test()

        else:
            obj.read_data()

            if obj._params["PSF_leakage"]:
                # Object-by-object spin-consistent PSF leakage
                obj.PSF_leakage()

            if obj._params["obs_leakage"]:
                # Object-by-object dependence of general variables
                obj.obs_leakage()


def run_leakage_object(*args):
    """Run Leakage Object.

    Run object-wise PSF leakage as python script from command line.

    """
    # Create object for object-wise leakage calculations
    obj = LeakageObject()

    obj.set_params_from_command_line(args)

    obj.run()
