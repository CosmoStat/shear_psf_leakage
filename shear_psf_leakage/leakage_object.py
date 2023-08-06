#!/usr/bin/env python3

import sys

from shear_psf_leakage.run_object import run_leakage_object


def main(argv=None):
    """Main

    Main program
    """
    if argv is None:
        argv = sys.argv[1:]
    run_leakage_object(*argv)

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
