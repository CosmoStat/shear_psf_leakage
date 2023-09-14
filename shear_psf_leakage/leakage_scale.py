#!/usr/bin/env python3

import sys

from shear_psf_leakage.run_scale import run_leakage_scale


def main(argv=None):
    """Main.

    Main program.

    """
    if argv is None:
        argv = sys.argv[0:]
    run_leakage_scale(*argv)

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
