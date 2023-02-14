import os
import sys

import subprocess as sp


def cif2patterson():
    from . import cif2patterson


def cif2topas():
    from . import cif2topas
    cif2topas.main()


def expandcell():
    from . import expandcell


def stripcif():
    from . import stripcif
    stripcif.main()


def topasdiff():
    from . import topasdiff
    topasdiff.main()


def make_superflip():
    from . import make_superflip
    make_superflip.main()


if __name__ == '__main__':
    print("Running...")
    print()
    topasdiff()
    print()
    print("Done!")
