from __future__ import print_function
from __future__ import absolute_import
import argparse
import os, sys

from .cif import reader, CifParserError


def read_cif(f):
    """opens cif and returns cctbx data object"""
    try:
        if isinstance(f, file):
            r = reader(file_object=f)
        elif isinstance(f, str):
            r = reader(file_path=f)
        else:
            raise TypeError('read_cif: Can not deal with type {}'.format(type(f)))
    except CifParserError as e:
        print(e)
        print("Error parsing cif file, check if the data tag does not contain any spaces.")
        sys.exit()
    structures = r.build_crystal_structures()
    for key, val in structures.items():
        print("\nstructure:", key)
        val.show_summary().show_scatterers()
    return structures


def main():
    description = """Notes:
    """

    epilog = 'Updated: {}'.format(__version__)

    parser = argparse.ArgumentParser(  # usage=usage,
        description=description,
        epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        version=__version__)

    parser.add_argument("args",
                        type=str, metavar="FILE",
                        help="Path to input cif")

    options = parser.parse_args()

    cif = options.args
    s = read_cif(cif).values()[0]

    # s = s.expand_to_p1()

    root, ext = os.path.splitext(cif)
    out = root + "_simple" + ext

    s.as_cif_simple(out=open(out, 'w'))
    print(" >> Wrote file {}".format(out))

if __name__ == '__main__':
    main()
