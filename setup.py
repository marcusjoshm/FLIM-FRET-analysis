#!/usr/bin/env python3
"""
Setup script for FLIM-FRET Analysis Pipeline package
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("src/scripts/requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="flimfret",
    version="1.0.0",
    author="Joshua Marcus",
    author_email="joshua.marcus@bcm.edu",
    description="FLIM-FRET Analysis Pipeline - Global Installation Available",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/marcusjoshm/FLIM-FRET-analysis",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Image Processing",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
        ],
    },
    entry_points={
        "console_scripts": [
            "flimfret=python.main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "python": [
            "config/*.json",
            "scripts/*.sh",
            "scripts/imagej/*.ijm",
        ],
    },
    zip_safe=False,
)
