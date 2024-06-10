"""LEAKAGE.

:Name: leakage.py

:Description: This package contains methods to deal with
    leakage.

:Authors: Martin Kilbinger <martin.kilbinger@cea.fr>
        Clara Bonini
        Axel Guinot

"""

import os

import numpy as np
import pickle
import matplotlib.pylab as plt
from lmfit import minimize, Parameters
from uncertainties import ufloat
from astropy.io import fits

from .plot_style import *


# MKDEBUG TODO: to cs_util (and see sp_validation/io.py)
def open_stats_file(directory, file_name):
    """Open statistics file.

    Open output file for statistics

    Parameters
    ----------
    directory : string
        directory
    file_name : string
        file name

    """
    stats_file = open("{}/{}".format(directory, file_name), "w")

    return stats_file

def print_stats(msg, stats_file, verbose=False):
    """Print stats.

    Print message to stats file.

    Parameters
    ----------
    msg : string
        message
    stats_file : file handler
        statistics output file
    verbose : bool, optional, default=False
        print message to stdout if True
    """
    stats_file.write(msg)
    stats_file.write("\n")
    stats_file.flush()

    if verbose:
        print(msg)


def open_fits_or_npy(path, hdu_no=1, verbose=False):
    """Open FITS OR NPY.

    Open FITS or numpy binary file.

    Parameters
    ----------
    path : str
        path to input binary file
    hdu_no : int, optional
        HDU number, default is 1
    verbose : bool, optional
        verbose output if ``True``; default is ``False``

    Raises
    ------
    ValueError
        if file extension not valid, i.e. neither ``.fits`` nor ``.npy``

    Returns
    -------
    FITS.rec or numpy.ndarray
        data

    """
    filename, file_extension = os.path.splitext(path)
    if file_extension in [".fits", ".cat"]:
        hdu_list = fits.open(path)
        data = hdu_list[hdu_no].data
    elif file_extension == ".npy":
        data = np.load(path)
    else:
        raise ValueError(f"Invalid file extension '{file_extension}'")

    if verbose:
        print(f"{len(data)} objects found in {file_extension} file")

    return data


def cut_data(data, cut, verbose=False):
    """Cut Data.

    Cut data according to selection criteria list.

    Parameters
    ----------
    data : numpy,ndarray
        input data
    cut : str
        selection criteria expressions, white-space separated
    verbose : bool, optional
        verbose output if `True`, default is `False`

    Raises
    ------
    ValueError :
        if cut expression is not valid

    Returns
    -------
    numpy.ndarray
        data after cuts

    """
    if cut is None:
        if verbose:
            print("No cuts applied to input galaxy catalogue")

        return data

    cut_list = cut.split(" ")

    for cut in cut_list:
        res = re.match(r"(\w+)([<>=!]+)(\w+)", cut)
        if res is None:
            raise ValueError(f"cut '{cut}' has incorrect syntax")
        if len(res.groups()) != 3:
            raise ValueError(
                f"cut criterium '{cut}' does not match syntax " "'field rel val'"
            )
        field, rel, val = res.groups()

        cond = "data['{}']{}{}".format(field, rel, val)

        if verbose:
            print(f"Applying cut '{cond}' to input galaxy catalogue")

        data = data[np.where(eval(cond))]

    if verbose:
        print(f"Using {len(data)} galaxies after cuts.")

    return data


def func_bias_2d_full(params, x1, x2, order="lin", mix=False):
    """Func Bias 2D Full.

    Function of 2D bias model evaluated on full 2D grid.

    Parameters
    ----------
    params : lmfit.Parameters
        fit parameters
    x1 : list
        first component of x-values, float
    x2 : list
        second component of x-values, float
    order : str, optional
        order of fit, default is 'lin'
    mix : bool, optional
        mixing between components, default is `False`

    Returns
    -------
    np.array
        first component the 2D model y1(x1, x2) on the (x1, x2)-grid;
        2D array of float
    np.array
        second component the 2D model, y2(x1, x2) on the (x1, x2)-grid;
        2D array of float

    """
    len1 = len(x1)
    len2 = len(x2)

    # Initialise both components y1, y2 as 2D arrays
    y1 = np.zeros(shape=(len1, len2))
    y2 = np.zeros(shape=(len1, len2))

    # Create 2D mesh for input x1, x2 values
    v1, v2 = np.meshgrid(x1, x2, indexing="ij")

    # Compute both components y1, y2 over the meash
    y1, y2 = func_bias_2d(params, v1, v2, order=order, mix=mix)

    return y1, y2


def func_bias_2d(params, x1_data, x2_data, order="lin", mix=False):
    """Func Bias 2D.

    Function of 2D bias model.

    Parameters
    ----------
    params : lmfit.Parameters
        fit parameters
    x1_data : float or list of float
        first component of x-values of the data
    x2_data : float or list of float
        second component of x-values of the data
    order : str, optional
        order of fit, default is 'lin'
    mix : bool, optional
        mixing between components, default is `False`

    Returns
    -------
    list
        first component the 2D model, y1(x1, x2). Dimension
        is equal to x1_data and x2_data
    list
        second component the 2D model, y2(x1, x2). Dimension
        is equal to x1_data and x2_data

    """
    # Get affine parameters
    a11 = params["a11"].value
    a22 = params["a22"].value
    c1 = params["c1"].value
    c2 = params["c2"].value

    # Compute y-values for affine model
    y1_model = a11 * x1_data + c1
    y2_model = a22 * x2_data + c2

    if order == "quad":
        # Add quadratic part
        q111 = params["q111"].value
        q222 = params["q222"].value
        y1_model += q111 * x1_data**2
        y2_model += q222 * x2_data**2

    if mix:
        # Add linear mixing part
        a12 = params["a12"].value
        a21 = params["a21"].value
        y1_model += a12 * x2_data
        y2_model += a21 * x1_data

        if order == "quad":
            # Add quadratic mixing part
            q112 = params["q112"].value
            q122 = params["q122"].value
            q212 = params["q212"].value
            q211 = params["q211"].value
            y1_model += q112 * x1_data * x2_data + q122 * x2_data**2
            y2_model += q212 * x1_data * x2_data + q211 * x1_data**2

    return y1_model, y2_model


def jackknife_mean_std(
    data,
    weights,
    remove_size=0.1,
    n_realization=100,
):
    """Jackknife Mean Standard Devitation.

    Computes weighted mean and standard deviation from jackknife resampling.

    Parameters
    ----------
    data : list
        input sample
    weights : list
        weights
    remove_size : float, optional
        fraction of input sample to remove for each jackknife resampling,
        default is ``0.1``
    n_realisation : int, optional
        number of jackknife resamples, default is ``100``

    Returns
    -------
    numpy.ndarray
        weighted mean
    numpy.ndarray
        weighted standard deviation


    """
    samp_size = len(data)
    keep_size_pc = 1 - remove_size

    if keep_size_pc < 0:
        raise ValueError("remove size should be in [0, 1]")

    subsamp_size = int(samp_size * keep_size_pc)

    all_ind = np.arange(samp_size)

    all_est = []
    for i in range(n_realization):
        sub_data_ind = np.random.choice(all_ind, subsamp_size)

        if sum(data[sub_data_ind]) == 0:
            all_est.append(np.nan)
        else:
            all_est.append(
                np.average(data[sub_data_ind], weights=weights[sub_data_ind])
            )

    all_est = np.array(all_est)

    return np.mean(all_est), np.std(all_est)


def func_bias_quad_1D(params, x_data):
    """Func Bias Quad 1D.

    Function for quadratic 1D bias model.

    Parameters
    ----------
    params : lmfit.Parameters
        fit parameters
    x_data : numpy.ndarray
        x-values of the data

    Returns
    -------
    numpy.ndarray
        y-values of the model

    """
    q = params["q"].value
    m = params["m"].value
    c = params["c"].value

    y_model = q * x_data**2 + m * x_data + c

    return y_model


def loss_bias_quad_1d(params, x_data, y_data, err):
    """Loss Bias quad 1D.

    Loss function for Quadratic 1D model

    Parameters
    ----------
    params : lmfit.Parameters
        fit parameters
    x_data : numpy.ndarray
        x-values of the data
    y_data : numpy.ndarray
        y-values of the data
    err : numpy.ndarray
        error values of the data

    Returns
    -------
    numpy.ndarray
        residuals

    """
    y_model = func_bias_quad_1D(params, x_data)
    residuals = (y_model - y_data) / err
    return residuals


def quad_corr_quant(
    x,
    y,
    xlabel,
    ylabel,
    qlabel=None,
    mlabel=None,
    clabel=None,
    weights=None,
    n_bin=30,
    out_path=None,
    title="",
    colors=None,
    stats_file=None,
    verbose=False,
    seed=None,
    rng=None,
):
    """Quadratic Correlation Quantity.

    Computes and plots quadratic correlation of y(n) as function of x.

    Parameters
    ----------
    x: array(double)
        input x value
    y: array(m) of double
        input y arrays
    xlabel, ylabel : str
        x-and y-axis labels
    mlabel : str, optional, default=None
        label for slope in the plot legend
    clabel : str, optional, default=None
        label for offset in the plot legend
    weights : array of double, optional, default=None
        weights of x points
    n_bin : double, optional, default=30
        number of points onto which data are binned
    out_path : str, optional, default=None
        output file path, if not given, plot is not saved to file
    title : str, optional, default=''
        plot title
    colors : array(m) of str, optional, default=None
        line colors
    stats_file : filehandler, optional, default=None
        output file for statistics
    verbose : bool, optional, default=False
        verbose output if True
    seed: int
        Seed to initialize the randoms. [Default: None]
    rng: numpy.random.RandomState
        Random generator. [Default: None]

    Returns
    -------
    list
        1rst order coeff of each e_gal vs quantities for recap plot
    list
        2nd order coeff of each e_gal vs quantities for recap plot
    list
        names of the quantities associated to each slopes
    list
        errors of 1rst order coeff
    list
        errors of 2nd order coeff

    """
    # Init randoms
    if isinstance(rng, np.random.RandomState):
        master_rng = rng
    else:
        master_rng = np.random.RandomState(seed)

    n_y = len(y)

    if qlabel is None:
        qlabel = np.full(n_y, "q")
    if mlabel is None:
        mlabel = np.full(n_y, "m")
    if clabel is None:
        clabel = np.full(n_y, "c")

    if weights is None:
        weights = np.ones_like(y[0])


    size_all = len(y[0])
    for idx in range(1, n_y):
        if len(y[idx]) != size_all:
            raise IndexError
            (
                f"Size {len(y[idx])} of input #{idx} is different from size "
                + f"{size_all} of input #0"
            )
    size_bin = int(size_all / n_bin)
    diff_size = size_all - size_bin

    # Prepare arrays for binned data
    x_arg_sort = np.argsort(x)
    x_bin = []
    y_bin = []
    err_bin = []

    for idx in range(len(y)):
        y_bin.append([])
        err_bin.append([])

    # Bin data for plot
    for idx in range(n_bin):
        if idx < diff_size:
            bin_size_tmp = size_bin + 1
            starter = 0
        else:
            bin_size_tmp = size_bin
            starter = diff_size
        ind = x_arg_sort[
            starter + idx * bin_size_tmp : starter + (idx + 1) * bin_size_tmp
        ]

        x_bin.append(np.mean(x[ind]))

        for j in range(len(y)):
            r_jk = jackknife_mean_std(
                y[j][ind],
                weights[ind],
                remove_size=0.2,
                n_realization=50,
            )
            y_bin[j].append(r_jk[0])
            err_bin[j].append(r_jk[1])

    x_bin = np.array(x_bin)
    for jdx in range(len(y)):
        y_bin[jdx] = np.array(y_bin[jdx])
        err_bin[jdx] = np.array(err_bin[jdx])

    # Fit affine functions, plot function and data
    slope = []
    qslope = []
    ticks_names = []
    m_err = []
    q_err = []
    plt.figure(figsize=(10, 6))
    for jdx in range(len(y)):
        params = Parameters()
        params.add("q", value=0.01)
        params.add("m", value=0.01)
        params.add("c", value=0.01)

        # Optimize parameters
        res = minimize(
            loss_bias_quad_1d, params, args=(x, y[jdx], 1 / np.sqrt(weights))
        )

        qslope.append(res.params["q"].value)
        slope.append(res.params["m"].value)

        ticks_names.append(f"{xlabel}_e_{jdx+1}")
        q_dm = ufloat(res.params["q"].value, res.params["q"].stderr)
        m_dm = ufloat(res.params["m"].value, res.params["m"].stderr)
        c_dc = ufloat(res.params["c"].value, res.params["c"].stderr)

        q_err.append(res.params["q"].stderr)
        m_err.append(res.params["m"].stderr)

        label = (
            rf"${qlabel[jdx]}={q_dm: .2ugL}, {mlabel[jdx]}={m_dm: .2ugL},"
            + f" {clabel[jdx]}={c_dc: .2ugL}$"
        )

        plt.plot(
            x_bin,
            func_bias_quad_1D(res.params, x_bin),
            c=colors[jdx],
            label=label,
        )

        plt.errorbar(
            x_bin,
            y_bin[jdx],
            yerr=err_bin[jdx],
            c=colors[jdx],
            fmt=".",
        )

        if stats_file:
            msg1 = "{}: {}={:.2ugP}".format(xlabel, qlabel[jdx], q_dm)
            msg2 = "{}: {}={:.2ugP}".format(xlabel, mlabel[jdx], m_dm)
            print_stats(msg1, stats_file, verbose=verbose)
            print_stats(msg2, stats_file, verbose=verbose)

    # Finalise plots
    plt_xmin, plt_xmax = plt.xlim()
    plt.xlim(plt_xmin, plt_xmax)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()

    plt.title(title)
    plt.tight_layout()

    if out_path:
        plt.savefig(out_path, bbox_inches="tight")
    plt.close()

    return slope, qslope, ticks_names, m_err, q_err


def quad_corr_n_quant(
    x_arr,
    y,
    xlabel_arr,
    ylabel,
    qlabel=None,
    mlabel=None,
    clabel=None,
    weights=None,
    n_bin=30,
    out_path_arr=None,
    title="",
    colors=None,
    stats_file=None,
    verbose=False,
    seed=None,
):
    """Quadratic Correlation N Quantity.

    Compute n quadratic correlations of y(m) versus x_arr[n].

    Parameters
    ----------
    x_arr: array(n, double)
        input x value
    y: array(m) of double
        input y arrays
    xlabel, ylabel : str
        x-and y-axis labels
    mlabel : str, optional, default=None
        label for slope in the plot legend
    clabel : str, optional, default=None
        label for offset in the plot legend
    weights : array of double, optional, default=None
        weights of x points
    n_bin : double, optional, default=30
        number of points onto which data are binned
    out_path_arr : array(n) of str, optional, default=None
        output file path, if not given, plot is not saved to file
    title : str, optional, default=''
        plot title
    colors(m) : array of str, optional, default=None
        line colors
    stats_file : filehandler, optional, default=None
        output file for statistics
    verbose : bool, optional, default=False
        verbose output if True
    seed: int
        Seed to initialize the randoms. [Default: None]

    """
    master_rng = np.random.RandomState(seed)
    seeds = master_rng.randint(low=0, high=2**30, size=len(x_arr))
    slopes = []
    qslopes = []
    ticks_label = []
    merr = []
    qerr = []

    if out_path_arr is None:
        out_path_arr = [None] * len(x_arr)
    for x, xlabel, out_path, seed_tmp in zip(x_arr, xlabel_arr, out_path_arr, seeds):
        slope, qslope, ticks_names, m_err, q_err = quad_corr_quant(
            x,
            y,
            xlabel,
            ylabel,
            mlabel=mlabel,
            clabel=clabel,
            weights=weights,
            n_bin=n_bin,
            out_path=out_path,
            title=title,
            colors=colors,
            stats_file=stats_file,
            verbose=verbose,
            seed=seed_tmp,
        )

        for i in range(len(slope)):
            slopes.append(slope[i])
            qslopes.append(qslope[i])
            ticks_label.append(ticks_names[i])
            merr.append(m_err[i])
            qerr.append(q_err[i])

    ticks_positions = np.arange(1, len(slopes) + 1, 1)

    # Plot slopes
    plt.figure()
    plt.errorbar(
        ticks_positions,
        slopes,
        yerr=merr,
        color="peru",
        label="m",
        fmt=".",
    )

    plt.errorbar(
        ticks_positions,
        qslopes,
        yerr=qerr,
        color="crimson",
        label="q",
        fmt=".",
    )

    plt.xticks(
        ticks_positions,
        ticks_label,
        rotation=90,
        fontsize=10,
    )

    plt.yticks(fontsize=10)
    plt.axhline(
        y=0,
        color="black",
        linestyle="--",
    )
    plt.ylabel("q and m", fontsize=10)
    title = "(e1, e2) systematic tests (quadratic)"
    plt.title(title, fontsize=10)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path_arr[-1])
    plt.close()


def func_bias_lin_1d(params, x_data):
    """Func Bias Lin 1D.

    Function for linear 1D bias model.

    Parameters
    ----------
    params : lmfit.Parameters
        fit parameters
    x_data : numpy.ndarray
        x-values of the data

    Returns
    -------
    numpy.ndarray
        y-values of the model

    """
    m = params["m"].value
    c = params["c"].value

    y_model = m * x_data + c

    return y_model


def loss_bias_lin_1d(params, x_data, y_data, err):
    """Loss Bias Lin 1D.

    Loss function for linear 1D model

    Parameters
    ----------
    params : lmfit.Parameters
        fit parameters
    x_data : numpy.ndarray
        x-values of the data
    y_data : numpy.ndarray
        y-values of the data
    err : numpy.ndarray
        error values of the data

    Returns
    -------
    numpy.ndarray
        residuals

    """
    y_model = func_bias_lin_1d(params, x_data)
    residuals = (y_model - y_data) / err
    return residuals


def loss_bias_2d(params, x_data, y_data, err, order, mix):
    """Loss Bias 2D.

    Loss function for 2D model

    Parameters
    ----------
    params : lmfit.Parameters
        fit parameters
    x_data : numpy.ndarray
        two-component x-values of the data
    y_data : numpy.ndarray
        two-component y-values of the data
    err : numpy.ndarray
        error values of the data, assumed the same for both components
    order : str
        order of fit
    mix : bool
        mixing of components if True

    Raises
    ------
    IndexError :
        if input arrays x1_data and x2_data have different lenght

    Returns
    -------
    numpy.ndarray
        residuals

    """
    # Get x and y values of the input data
    x1_data = x_data[0]
    x2_data = x_data[1]
    y1_data = y_data[0]
    y2_data = y_data[1]

    if len(x1_data) != len(x2_data):
        raise IndexError("Length of both data components has to be equal")

    # Get model 1D y1 and y2 components
    y1_model, y2_model = func_bias_2d(
        params, x1_data, x2_data, order=order, mix=mix
    )

    # Compute residuals between data and model
    res1 = (y1_model - y1_data) / err
    res2 = (y2_model - y2_data) / err

    # Concatenate both components
    residuals = np.concatenate([res1, res2])

    return residuals


def print_fit_report(res, file=None):
    """Print Fit Report.

    Print report of minimizing result.

    Parameters
    ----------
    res : class lmfit.MinimizerResult
        results of the minization
    file : filehandler, optional
        output to file; if `None` (default) output to `stdout`

    """
    # chi^2
    print(f"chi^2 = {res.chisqr}", file=file)

    # Reduced chi^2
    print(f"reduced chi^2 = {res.redchi}", file=file)

    # Akaike Information Criterium
    print(f"aic = {res.aic}", file=file)

    # Bayesian Information Criterium
    print(f"bic = {res.bic}", file=file)


def corr_2d(
    x,
    y,
    weights=None,
    order="lin",
    mix=False,
    stats_file=None,
    verbose=False,
):
    """Corr 2D.

    Compute and plot 2D linear and quadratic correlations of (y1, y2) as
    function of (x1, x2).

    Parameters
    ----------
    x : array(double)
        input x value
    y : array(m) of double
        input y arrays
    weights  : array of double, optional, default=None
        weights of x points
    order : str, optional
        order of fit, default is 'lin'
    mix : bool
        mixing of components if True
    stats_file : filehandler, optional, default=None
        output file for statistics
    verbose : bool, optional
        verbose output if ``True``; default is ``False``

    Returns
    -------
    lmfit.Parameters
        best-fit parameters

    """
    if len(y) != 2 or len(x) != 2:
        raise IndexError("Input data needs to have two components")
    if any(len(y[0]) != c for c in {len(y[1]), len(x[0]), len(x[1])}):
        raise IndexError("Input data has inconsistent length")

    # Initialise parameters of model to fit
    params = Parameters()

    val_init = 0.0

    # Affine parameters
    for p_affine in ["a11", "a22", "c1", "c2"]:
        params.add(p_affine, value=val_init)

    if mix:
        # Linear mixing pararmeters
        params.add("a12", value=val_init)
        params.add("a21", value=val_init)

    if order == "quad":
        # Quadratic parameters
        for p_quad in ["q111", "q222"]:
            params.add(p_quad, value=val_init)

        if mix:
            # Quadratic mixing parameters
            for p_quad_mix in ["q112", "q122", "q212", "q211"]:
                params.add(p_quad_mix, value=val_init)

    # Mininise loss function
    err = 1 / np.sqrt(weights) if weights is not None else np.ones_like(y[0])
    res = minimize(loss_bias_2d, params, args=(x, y, err, order, mix))
    if stats_file:
        print_stats(
            f"2D fit order={order} mix={mix}:",
            stats_file,
            verbose=verbose,
        )
        print_fit_report(res, file=stats_file)
    if verbose:
        print_fit_report(res)

    return res.params


def affine_corr(
    x,
    y,
    xlabel,
    ylabel,
    mlabel=None,
    clabel=None,
    weights=None,
    n_bin=30,
    out_path=None,
    title="",
    colors=None,
    stats_file=None,
    verbose=False,
    seed=None,
    rng=None,
):
    """Affine Corr.

    Computes and plots affine correlation of y(n) as function of x.

    Parameters
    ----------
    x: array(double)
        input x value
    y: array(m) of double
        input y arrays
    xlabel, ylabel : str
        x-and y-axis labels
    mlabel : str, optional, default=None
        label for slope in the plot legend
    clabel : str, optional, default=None
        label for offset in the plot legend
    weights : array of double, optional, default=None
        weights of x points
    n_bin : double, optional, default=30
        number of points onto which data are binned
    out_path : str, optional, default=None
        output file path, if not given, plot is not saved to file
    title : str, optional, default=''
        plot title
    colors : array(m) of str, optional, default=None
        line colors
    stats_file : filehandler, optional, default=None
        output file for statistics
    verbose : bool, optional, default=False
        verbose output if True
    seed: int
        Seed to initialize the randoms. [Default: None]
    rng: numpy.random.RandomState
        Random generator. [Default: None]

    Returns
    -------
    list
        slopes of the linear fits
    list
        errors of the slopes
    list
        labels of the linear fits

    """
    # Init randoms
    if isinstance(rng, np.random.RandomState):
        master_rng = rng
    else:
        master_rng = np.random.RandomState(seed)

    n_y = len(y)

    if mlabel is None:
        mlabel = np.full(n_y, r"\alpha")
    if clabel is None:
        clabel = np.full(n_y, "c")

    if weights is None:
        weights = np.ones_like(y[0])

    if colors is None:
        prop_cycle = plt.rcParams["axes.prop_cycle"]
        colors = prop_cycle.by_key()["color"]

    size_all = len(y[0])
    for idx in range(1, n_y):
        if len(y[idx]) != size_all:
            raise IndexError
            (
                f"Size {len(y[idx])} of input #{idx} is different from size "
                + f"{size_all} of input #0"
            )
    size_bin = int(size_all / n_bin)
    diff_size = size_all - size_bin

    # Prepare arrays for binned data
    x_arg_sort = np.argsort(x)
    x_bin = []
    y_bin = []
    err_bin = []

    for idx in range(len(y)):
        y_bin.append([])
        err_bin.append([])

    # Bin data for plot
    for idx in range(n_bin):
        if idx < diff_size:
            bin_size_tmp = size_bin + 1
            starter = 0
        else:
            bin_size_tmp = size_bin
            starter = diff_size
        ind = x_arg_sort[
            starter + idx * bin_size_tmp : starter + (idx + 1) * bin_size_tmp
        ]

        x_bin.append(np.mean(x[ind]))

        for j in range(len(y)):
            r_jk = jackknife_mean_std(
                y[j][ind],
                weights[ind],
                remove_size=0.2,
                n_realization=50,
            )
            y_bin[j].append(r_jk[0])
            err_bin[j].append(r_jk[1])

    x_bin = np.array(x_bin)
    for jdx in range(len(y)):
        y_bin[jdx] = np.array(y_bin[jdx])
        err_bin[jdx] = np.array(err_bin[jdx])

    # Fit affine functions, plot function and data
    plt.figure(figsize=(10, 6))

    m_arr = []
    m_err_arr = []
    tick_name_arr = []

    for jdx in range(len(y)):
        params = Parameters()
        params.add("m", value=0.01)
        params.add("c", value=0.01)
        res = minimize(loss_bias_lin_1d, params, args=(x, y[jdx], 1 / np.sqrt(weights)))

        m_arr.append(res.params["m"].value)
        # MKDEBUG float required?
        m_err_arr.append(float(res.params["m"].stderr))
        tick_name_arr.append(f"{xlabel}_e{jdx+1}")

        m_dm = ufloat(res.params["m"].value, res.params["m"].stderr)
        c_dc = ufloat(res.params["c"].value, res.params["c"].stderr)
        label = rf"${mlabel[jdx]}={m_dm: .2ugL}, {clabel[jdx]}={c_dc: .2ugL}$"
        plt.plot(x_bin, func_bias_lin_1d(res.params, x_bin), c=colors[jdx], label=label)
        plt.errorbar(x_bin, y_bin[jdx], yerr=err_bin[jdx], c=colors[jdx], fmt=".")

        if stats_file:
            msg = "{}: {}={:.2ugP}".format(xlabel, mlabel[jdx], m_dm)
            print_stats(msg, stats_file, verbose=verbose)

    # Finalise plots
    plt_xmin, plt_xmax = plt.xlim()
    plt.xlim(plt_xmin, plt_xmax)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()

    plt.title(title)
    plt.tight_layout()

    if out_path:
        plt.savefig(out_path, bbox_inches="tight")

    plt.close()

    return m_arr, m_err_arr, tick_name_arr


def affine_corr_n(
    x_arr,
    y,
    xlabel_arr,
    ylabel,
    mlabel=None,
    clabel=None,
    weights=None,
    n_bin=30,
    out_path_arr=None,
    title="",
    colors=None,
    stats_file=None,
    verbose=False,
    seed=None,
):
    """Affine Corr N.

    Compute n affine correlations of y(m) versus x_arr[n].

    Parameters
    ----------
    x_arr: array(n, double)
        input x value
    y: array(m) of double
        input y arrays
    xlabel, ylabel : str
        x-and y-axis labels
    mlabel : str, optional, default=None
        label for slope in the plot legend
    clabel : str, optional, default=None
        label for offset in the plot legend
    weights : array of double, optional, default=None
        weights of x points
    n_bin : double, optional, default=30
        number of points onto which data are binned
    out_path_arr : array(n) of str, optional, default=None
        output file path, if not given, plot is not saved to file
    title : str, optional, default=''
        plot title
    colors(m) : array of str, optional, default=None
        line colors
    stats_file : filehandler, optional, default=None
        output file for statistics
    verbose : bool, optional, default=False
        verbose output if True
    seed: int
        Seed to initialize the randoms. [Default: None]

    """
    master_rng = np.random.RandomState(seed)
    seeds = master_rng.randint(low=0, high=2**30, size=len(x_arr))

    if out_path_arr is None:
        out_path_arr = [None] * len(x_arr)
    for x, xlabel, out_path, seed_tmp in zip(x_arr, xlabel_arr, out_path_arr, seeds):
        m_arr, m_err_arr, tick_name_arr = affine_corr(
            x,
            y,
            xlabel,
            ylabel,
            mlabel=mlabel,
            clabel=clabel,
            weights=weights,
            n_bin=n_bin,
            out_path=out_path,
            title=title,
            colors=colors,
            stats_file=stats_file,
            verbose=verbose,
            seed=seed_tmp,
        )

    # Summary plot
    plt.figure()
    ticks_positions = np.arange(1, len(m_arr) + 1, 1)
    plt.errorbar(ticks_positions, m_arr, yerr=m_err_arr, color="peru", fmt=".")
    plt.xticks(
        ticks_positions,
        tick_name_arr,
        rotation=90,
        fontsize=10,
    )
    plt.yticks(fontsize=10)
    plt.axhline(
        y=0,
        color="black",
        linestyle="--",
    )
    plt.ylabel("m")
    title = "(e1, e2) systematic tests"
    plt.title(title, fontsize=10)
    plt_xmin, plt_xmax = plt.xlim()
    plt.xlim(plt_xmin, plt_xmax)
    plt.tight_layout()
    plt.savefig(out_path_arr[-1])
    plt.close()


def save_to_file(data, fname):
    """Save To File.

    Save data to .pkl (pickle) file.

    Parameters
    ----------
    data : dict
        input data
    fname : str
        output file name

    See also
    --------
    read_from_file

    """
    with open(fname, "wb") as f:
        pickle.dump(data, f)


def read_from_file(fname):
    """Read From File.

    Read data from .pkl (pickle) file.

    Parameters
    ----------
    fname : str
        input file name

    Returns
    -------
    dict
        data

    See also
    --------
    save_to_file

    """

    with open(fname, "rb") as f:
        data = pickle.load(f)

    return data


def param_order2spin(p_dp, order, mix):
    """Param Order 2 Spin.

    Transform parameter from natural to spin coefficients.

    Parameters
    ----------
    p_dp : dict
        Parameter natural coefficients
    order : str
        expansion order, one of 'linear', 'quad'
    mix : bool
        ellipticity components are mixed if ``True``

    Returns
    -------
    dict
        Parameter spin coefficients

    """
    s_ds = {"x0": 0.5 * (p_dp["a11"] + p_dp["a22"])}

    if order == "quad" and mix:
        s_ds["x2"] = 0.5 * (p_dp["q111"] + p_dp["q122"])
        s_ds["y2"] = 0.5 * (p_dp["q211"] - p_dp["q222"])
        s_ds["x-2"] = 0.25 * (p_dp["q111"] - p_dp["q122"] + p_dp["q212"])
        s_ds["y-2"] = 0.25 * (p_dp["q211"] - p_dp["q222"] - p_dp["q112"])

    s_ds["x4"] = 0.5 * (p_dp["a11"] - p_dp["a22"])

    if mix:
        s_ds["y4"] = 0.5 * (p_dp["a12"] + p_dp["a21"])
        s_ds["y0"] = 0.5 * (-p_dp["a12"] + p_dp["a21"])

    if order == "quad" and mix:
        s_ds["x6"] = 0.25 * (p_dp["q111"] - p_dp["q122"] - p_dp["q212"])
        s_ds["y6"] = 0.25 * (p_dp["q211"] - p_dp["q222"] + p_dp["q112"])

    return s_ds

