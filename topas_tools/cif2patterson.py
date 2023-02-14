import argparse

import os, sys
import io
from iotbx.cif import reader, CifParserError


def read_cif(f):
    """opens cif and returns cctbx data object"""
    try:
        if isinstance(f, io.IOBase):
            structures = reader(file_object=f).build_crystal_structures()
        elif isinstance(f, str):
            structures = reader(file_path=f).build_crystal_structures()
        else:
            raise TypeError(f'read_cif: Can not deal with type {type(f)}')
    except CifParserError as e:
        print(e)
        print("Error parsing cif file, check if the data tag does not contain any spaces.")
        sys.exit()
    for key, val in list(structures.items()):
        print("\nstructure:", key)
        val.show_summary().show_scatterers()
    return structures


usage = """cif2patterson structure.cif"""

description = """Notes: Takes any cif file and generated patterson map
"""

parser = argparse.ArgumentParser(  # usage=usage,
    description=description,
    formatter_class=argparse.RawDescriptionHelpFormatter)


parser.add_argument("args",
                    type=str, metavar="FILE",
                    help="Path to input cif")


parser.set_defaults(
    spgr="P1"
)

options = parser.parse_args()

cif = options.args
s = list(read_cif(cif).values())[0]
s = s.expand_to_p1()
print(f"Expanded to P1 => {s.scatterers().size()} atoms")
print()

root, ext = os.path.splitext(cif)

uc = s.unit_cell()

scatterers = s.scatterers()

fout1 = open("patterson_dists.txt", 'w')
fout2 = open("patterson_full.txt", 'w')

distances_all = []

verbose = False
for atom1 in scatterers:
    x1, y1, z1 = atom1.site

    if verbose:
        print()
        atom1.show()
    distances = []
    for atom2 in scatterers:
        if verbose:
            atom2.show()

        x2, y2, z2 = atom2.site

        dx, dy, dz = x1-x2, y1-y2, z1-z2

        dx = dx % 1
        dy = dy % 1
        dz = dz % 1

        length = uc.length((dx, dy, dz))

        distances.append((atom2.label, dx, dy, dz, length))

        if verbose:
            print(f' --> {atom2.label:>4s} {dx:9.5f} {dy:9.5f} {dz:9.5f}  {length:9.5f}')

        # print atom1.label, '-->', atom2.label, '=', uc.length((dx,dy,dz))

    distances_all.extend(distances)

    atom1.show(fout2)
    for label, dx, dy, dz, distance in sorted(distances, key=lambda x: x[-1]):
        print(' --> {:>4s} {:9.5f} {:9.5f} {:9.5f}  {:9.5f}'.format(
            label, dx, dy, dz, distance), file=fout2)
    print(file=fout2)

print('Wrote file', fout2.name)

for label, dx, dy, dz, distance in sorted(distances_all, key=lambda x: x[-1]):
    print(f'{distance:9.5f}', file=fout1)

print('Wrote file', fout1.name)

fout1.close()
fout2.close()

sys.exit()