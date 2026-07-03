"""PLOT STYLE.

:Name: plot_style.py

:Description: Optional matplotlib plot style for shear_psf_leakage. Call
              ``set_style()`` to apply it. Importing this module no longer
              mutates matplotlib's global rcParams as a side effect -- that
              silently restyled any code that imported shear_psf_leakage
              (e.g. sp_validation, whose cosmo_val imports the leakage
              utilities), which was surprising. Applying a style is now
              opt-in and explicit.

:Author: Axel Guinot

"""

import matplotlib as mpl

font_size = 18


def set_style():
    """Set Style.

    Apply the shear_psf_leakage matplotlib plot style (opt-in). Mutates the
    global ``matplotlib.rcParams``, so call it from a script/notebook when you
    want this look -- not at import time.

    """
    mpl.rcParams["lines.linewidth"] = 2
    mpl.rcParams["lines.markersize"] = 10

    mpl.rcParams["font.size"] = font_size
    mpl.rcParams["xtick.labelsize"] = font_size
    mpl.rcParams["ytick.labelsize"] = font_size

    mpl.rcParams["xtick.minor.size"] = 5
    mpl.rcParams["ytick.minor.size"] = 5

    mpl.rcParams["xtick.major.size"] = 7
    mpl.rcParams["ytick.major.size"] = 7

    mpl.rcParams["xtick.major.width"] = 2
    mpl.rcParams["ytick.major.width"] = 2

    mpl.rcParams["boxplot.boxprops.linewidth"] = 2
    mpl.rcParams["boxplot.medianprops.linewidth"] = 2
    mpl.rcParams["boxplot.flierprops.markersize"] = 12
    mpl.rcParams["boxplot.whiskerprops.linewidth"] = 2
    mpl.rcParams["boxplot.capprops.linewidth"] = 2

    mpl.rcParams["axes.xmargin"] = mpl.rcParamsDefault["axes.xmargin"]
