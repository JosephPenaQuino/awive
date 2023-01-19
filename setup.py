"""Setup file."""
from setuptools import setup, find_packages


setup(
    name="adaptive-water-image-velocimetry-estimator",
    version="0.1.dev0",
    license="",
    long_description=open("README.md").read(),
    description="A python package for estimating the velocity of water images",
    url=(
        "https://github.com/JosephPenaQuino/"
        "adaptive-water-image-velocimetry-estimator"
    ),
    author='Joseph Pena',
    author_email='joseph.pena@utec.edu.pe',
    packages=find_packages(),
    install_requires=[
    ],

    classifiers=[
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3.9',
    ],
)
