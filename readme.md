# Topas_tools

These are set of small scripts and tools that were developed to help with structure refinement (of zeolites in particular) using the program TOPAS (http://topas-academic.net/).

## topasdiff

Topasdiff is a tool to generate nice looking difference maps. First, output the observed structure factors, and structure in cif format from Topas:

    Out_fobs(fobs.hkl)
    Out_CIF_STR(structure.cif)

Observed structure factors can be generated using this macro (add this to C:/topas5/local.inc):

    macro Out_fobs(file)
    {
       phase_out file load out_record out_fmt out_eqn
       {
           "%4.0f" = H;
           "%4.0f" = K;
           "%4.0f" = L;
           '"%4.0f" = M;
           "%12.4f\n" = (Iobs_no_scale_pks / M)^(0.5);
       }
    }

Then to calculate the difference map, run the command:

    topasdiff structure.cif --diff fobs.hkl

or use the GUI, accessible available via topasdiff.bat (Windows). The program will ask for the scale factor provided by TOPAS, and generates an input file for the program superflip (http://superflip.fzu.cz/). Superflip is then used to perform the fourier transform, and generates an XPLOR file that can be visualized using Chimera or VESTA.

Note: There is a bug in Topas where the cif files it outputs cannot be read using CCTBX, because they lack the data header. A work-around is to open C:/topas5/topas.inc and modify all instances of:

    Out_String("\ndata_")

by:
   
    Out_String("\ndata_topas_cif_out")

![topasdiff gui](https://cloud.githubusercontent.com/assets/873520/14959028/c68ba2e4-108d-11e6-9942-f8e6acc1559f.png)

## cif2topas

cif2topas transforms a crystal structure into cif format into the corresponding TOPAS code.

Usage:

    cif2topas structure.cif


## fh2topas

fh2topas is a script that converts from Fenske-Hall z-matrix to TOPAS code

Usage:

    fh2topas.py zmatrix.fh [n]

Here, is an optional parameter to give the number of molecules to generate. fh2topas will automatically number them to avoid naming conflicts.


## topasrestraints

topasrestraints is a tool that can help with the generation of bond and angle restraints for structure refinement of zeolites with TOPAS. First, generate all bonds by using the TOPAS instruction:

    append_bond_lengths

to generate all bonds and angles, including their symmetry codes for the current structure. Copy everything between the curly brackets to a new file called bonds.txt, and run:

    topasrestraints bonds.txt

This generates a file called restraints.out that contains the restraints that can be added to TOPAS. The script checks for all distances of 1.61 +- 0.4 Angstrom to identify T-O bonds. The program checks the connectivity for every atom, and reports if there is a problem (too many / not enough distances per T-atom). There is no check for symmetrically equivalent connections, so some restraints may be redundant.


## Requirements

- Python2.7
- numpy
- matplotlib
- pandas
- CCTBX
- superflip (http://superflip.fzu.cz/)


## Installation

Download and extract:

https://github.com/stefsmeets/topas_tools/archive/master.zip

Install:

    python setup.py install

Uninstall:

    pip uninstall topas_tools
