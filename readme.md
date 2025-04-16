[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/topas_tools)](https://pypi.org/project/topas_tools/)
[![PyPI](https://img.shields.io/pypi/v/topas_tools.svg?style=flat)](https://pypi.org/project/topas_tools/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4719228.svg)](https://doi.org/10.5281/zenodo.4719228)

# Topas tools

These are set of small scripts and tools that were developed to help with structure refinement (of zeolites in particular) using the program [TOPAS](http://topas-academic.net/).

## topasdiff

Topasdiff is a tool to generate nice looking difference maps. First, output the observed structure factors, and structure in cif format from Topas:

    Out_fobs(fobs.hkl)
    Out_CIF_STR(structure.cif)

Observed structure factors can be generated using this macro (add this to `C:/topas5/local.inc`):

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

or use the GUI, accessible available via `topasdiff.bat` (Windows). The program will ask for the scale factor provided by TOPAS, and generates an input file for the program superflip (http://superflip.fzu.cz/). Superflip is then used to perform the fourier transform, and generates an `.XPLOR` file that can be visualized using Chimera or VESTA.

Note: There is a bug in Topas where the cif files it outputs cannot be read using CCTBX, because they lack the data header. A work-around is to open `C:/topas5/topas.inc` and modify all instances of:

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

    fh2topas zmatrix.fh [n]

Here, `n` is an optional parameter to give the number of molecules to generate. fh2topas will automatically number them to avoid naming conflicts.


## topasrestraints

topasrestraints is a tool that can help with the generation of bond and angle restraints for structure refinement of zeolites with TOPAS. First, generate all bonds by using the TOPAS instruction:

    append_bond_lengths

to generate all bonds and angles, including their symmetry codes for the current structure. Copy everything between the curly brackets to a new file called `bonds.txt`, and run:

    topasrestraints bonds.txt

This generates a file called restraints.out that contains the restraints that can be added to TOPAS. The script checks for all distances of `1.61 +- 0.4` Angstrom to identify T-O bonds. The program checks the connectivity for every atom, and reports if there is a problem (too many / not enough distances per T-atom). There is no check for symmetrically equivalent connections, so some restraints may be redundant.

## stripcif

stripcif is a tool to clean a cif file from all extra metadata. Essentially it reads a cif file and writes it back to `structure_simple.cif` with just the cell parameters and atomic coordinates.

Usage:

    stripcif structure.cif

## expandcell

expandcell is a tool to take a cif file and expand the axes along different directions to form a supercell.

For example:

    expandcell structure.cif -z2

will write a cif in `P1` the *z* axis doubled. You can then use a tool like [PLATON](http://www.platonsoft.nl/platon/pl000000.html) to find the right symmetry for this structure (if needed). You can use:

    expandcell structure.cif -z2 -s SPGR --shift X Y Z

to force the new a new symmetry on the output. Here `SPGR` is the space group suggested by platon, and `--shift X Y Z` is the suggested origin shift.

The option `-e` can be used to exclude elements from the result.

## make_superflip

This is a very simple tool that asks a couple of questions and writes an input file for superflip.

Usage:

    make_superflip

## Requirements

- Python>=3.9
- numpy
- matplotlib
- pandas
- [CCTBX](https://github.com/cctbx/cctbx_project)
- superflip ([superflip.fzu.cz/](http://superflip.fzu.cz/))

## Installation

1. Install [miniconda](https://docs.conda.io/en/latest/miniconda.html) (pick the one suitable for your platform, Python version does not matter here).

2. Install using:

```bash
conda create -n topas_tools -c conda-forge cctbx-base
conda activate topas_tools
pip install topas_tools
```

Or use the environment file:

```bash
conda env create -f environment.yml
conda activate topas_tools
pip install -e .
```

(note that every time you want to use `topas_tools`, you must always activate the environment using `conda activate topas_tools`)

## How to Cite

If you find this software useful, please consider citing it:

- Smeets, S. (2021). topas_tools (Version 1.1.0) [Computer software]. https://doi.org/10.5281/zenodo.4719229
