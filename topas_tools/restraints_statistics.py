import os
import re
import sys


def pbpaste():
    """Mac only: paste text from clipboard"""
    outf = os.popen('pbpaste', 'r')
    content = outf.read()
    outf.close()
    return content


def equals_about(val, compare_to):
    return compare_to * 0.9 <= val <= compare_to*1.1


def main():
    if len(sys.argv) > 1:
        text = open(sys.argv[1]).read()
    else:
        if sys.platform == "darwin":
            text = pbpaste()
        else:
            print('Usage: python restraints_statistics.py [filename]')
            sys.exit()

    oto = 109.5
    tot = 145.0
    to = 1.61

    oto_vals = []
    tot_vals = []
    to_vals = []

    n_lines = 0
    n_oto = 0
    n_tot = 0
    n_to = 0

    for line in re.split('\r|\n', text):
        n_lines += 1

        line = line.strip()

        if line.startswith("'"):
            continue
        if "Restrain" not in line:
            continue
        if "Si" not in line:
            continue

        stuff, restraint, measured, box, weight = line.split(',')

        restraint = float(restraint)
        measured = float(measured.replace('`', '').split("_")[0])

        if equals_about(restraint, oto):
            oto_vals.append(measured)
            n_oto += 1
        elif equals_about(restraint, tot):
            tot_vals.append(measured)
            n_tot += 1
        elif equals_about(restraint, to):
            to_vals.append(measured)
            n_to += 1
        else:
            print(line + ' -- FAIL')

    if not tot_vals:
        tot_vals = [0]
    if not to_vals:
        to_vals = [0]
    if not oto_vals:
        oto_vals = [0]

    print(f"Parsed {n_lines} lines")
    print(f"{n_tot+n_oto+n_to} restraints - tot: {n_tot}, oto: {n_oto}, to: {n_to}")
    print("")
    print('        {:>10s} {:>10s} {:>10s} {:>10s} '.format('restraint', 'min', 'max', 'avg'))
    print(f' T-O-T  {tot:10.1f} {min(tot_vals):10.3f} {max(tot_vals):10.3f} {sum(tot_vals)/len(tot_vals):10.3f} ')
    print(f' O-T-O  {oto:10.1f} {min(oto_vals):10.3f} {max(oto_vals):10.3f} {sum(oto_vals)/len(oto_vals):10.3f} ')
    print(f'   T-O  {to:10.2f} {min(to_vals):10.3f} {max(to_vals):10.3f} {sum(to_vals)/len(to_vals):10.3f} ')

if __name__ == '__main__':
    main()
