import glob
from setuptools import setup, find_packages

with open("requirements.txt") as f:
    required = f.read().splitlines()
    
setup(
    name="polixrspgen",
    version="0.1.0",
    packages=find_packages(),
    url="https://github.com/sujaymate/polixrspgen",
    author="Sujay",
    scripts=glob.glob("bin/*"),
    install_requires=required,
    author_email="sujay.mate@gmail.com",
    zip_safe=False,
    description="Package to generate response matrices for XPoSat/POLIX",
    classifiers=[
        "Natural Language :: English",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Topic :: Scientific/Engineering :: Astronomy",
    ])