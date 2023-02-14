import re
import sys


def main():
    pat = re.compile('[a-zA-Z]+')

    if len(sys.argv) == 1:
        print("Usage: fh2topas.py ~/path/to/output.fh [n]")
        sys.exit()

    args = sys.argv[2:]

    moltotal = 1
    if args:
        moltotal = int(args[0])
    molnum = 1

    fin = open(sys.argv[1])

    lines = fin.readlines()

    molnum = 1
    while molnum <= moltotal:
        if moltotal == 1:
            molnum = ''

        number = None
        n = 0
        d = {}
        all_atoms = []

        print('      rigid')

        for line in lines:
            inp = line.split()

            if not inp:
                continue
            if not number:
                number = int(inp[0])
                continue
            n += 1

            print('         z_matrix ', end=' ')

            if n > 0:
                tpe = inp[0]

                atom = f'{tpe}{n}'
                d[str(n)] = atom
                all_atoms.append(atom)

                scatterer = re.findall(pat, atom)[0]
                atom = atom.replace(scatterer, scatterer+str(molnum))
                print(f'{atom:6s}', end=' ')

            if n > 1:
                bonded_with, bond = inp[1:3]
                bonded_with = d[bonded_with]
                scatterer = re.findall(pat, bonded_with)[0]
                bonded_with = bonded_with.replace(
                    scatterer, scatterer+str(molnum))
                print(f'{bonded_with:6s} {bond:7s}', end=' ')
            if n > 2:
                angle_with, angle = inp[3:5]
                angle_with = d[angle_with]
                scatterer = re.findall(pat, angle_with)[0]
                angle_with = angle_with.replace(
                    scatterer, scatterer+str(molnum))
                print(f'{angle_with:6s} {angle:7s}', end=' ')
            if n > 3:
                torsion_with, torsion = inp[5:7]
                torsion_with = d[torsion_with]
                scatterer = re.findall(pat, torsion_with)[0]
                torsion_with = torsion_with.replace(
                    scatterer, scatterer+str(molnum))
                print(f'{torsion_with:6s} {torsion:>7s}', end=' ')

            print()

        print("""
             Rotate_about_axies(@ 0.0 randomize_on_errors,
                                @ 0.0 randomize_on_errors,
                                @ 0.0 randomize_on_errors)
             Translate(         @ 0.0 randomize_on_errors,
                                @ 0.0 randomize_on_errors,
                                @ 0.0 randomize_on_errors)
        """)

        if molnum == '':
            break
        else:
            molnum += 1

    molnum = 1
    while molnum <= moltotal:
        if moltotal == 1:
            molstr = ''
        else:
            molstr = str(molnum)
        print()
        print(f"      prm !occ{molnum} 0.5 min 0.0 max  1.0")
        print(f"      prm !beq{molnum} 3.0 min 1.0 max 10.0")
        print()

        for atom in all_atoms:
            scatterer = re.findall(pat, atom)[0]
            atom = atom.replace(scatterer, scatterer+molstr)
            print(
                f'      site {atom:6s} x 0.0 y 0.0 z 0.0 occ {scatterer:2s} =occ{molnum}; beq =beq{molnum};')
        molnum += 1


if __name__ == '__main__':
    main()
