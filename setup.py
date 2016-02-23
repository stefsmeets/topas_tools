#!/usr/bin/env python

from setuptools import setup, find_packages
from os import path

setup(
    name="topas_tools",
    version="0.1.2",
    description="Set of tools to aid structure refinement with TOPAS",

    author="Stef Smeets",
    author_email="stef.smeets@mmk.su.se",
    license="GPL",
    url="https://github.com/stefsmeets/topas_tools",

    classifiers=[
        'Programming Language :: Python :: 2.7',
    ],

    packages=["topas_tools"],

    install_requires=["numpy", "pandas", "matplotlib"],

    package_data={
        "": ["LICENCE", "readme.md"],
        "topas_tools": ["*.py"]
    },

    entry_points={
        'console_scripts': [
            'fh2topas = topas_tools.fh2topas:main',
            'topasrestraints = topas_tools.topas_restraints:main',
            'topas_ndxan = topas_tools.topas_ndxan:main',
            'restraints_statistics = topas_tools.restraints_statistics:main',
            'cif2patterson = topas_tools.__main__:cif2patterson',
            'cif2topas = topas_tools.__main__:cif2topas',
            'expandcell = topas_tools.__main__:expandcell',
            'stripcif = topas_tools.__main__:stripcif',
            'topasdiff = topas_tools.__main__:topasdiff',
            'make_superflip = topas_tools.__main__:make_superflip',
        ]
    }

)
