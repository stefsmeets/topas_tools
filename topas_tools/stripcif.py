import argparse
import os, sys
import io
from iotbx.cif import reader, CifParserError


def read_cif(f):
    """opens cif and returns cctbx data object"""
    try:
        if isinstance(f, io.IOBase):
            r = reader(file_object=f)
        elif isinstance(f, str):
            r = reader(file_path=f)
        else:
            raise TypeError(f'read_cif: Can not deal with type {type(f)}')
    except CifParserError as e:
        print(e)
        print("Error parsing cif file, check if the data tag does not contain any spaces.")
        sys.exit()
    structures = r.build_crystal_structures()
    for key, val in list(structures.items()):
        print("\nstructure:", key)
        val.show_summary().show_scatterers()
    return structures


def main():
    description = """Notes:
    """

    parser = argparse.ArgumentParser(  # usage=usage,
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("args",
                        type=str, metavar="FILE",
                        help="Path to input cif")

    options = parser.parse_args()

    cif = options.args
    s = list(read_cif(cif).values())[0]

    # s = s.expand_to_p1()

    root, ext = os.path.splitext(cif)
    out = root + "_simple" + ext

    s.as_cif_simple(out=open(out, 'w'))
    print(f" >> Wrote file {out}")

if __name__ == '__main__':
    main()
