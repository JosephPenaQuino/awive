"""Setup file."""
from setuptools import setup, find_packages


setup(
    name="adaptive-water-image-velocimetr-estimator",
    version="0.1dev",
    license="",
    long_description=open('README').read(),
    description="A python package for estimating the velocity of water images",
    url=(
        "https://github.com/JosephPenaQuino/"
        "adaptive-water-image-velocimetry-estimator"
    ),
    author='omners',
    author_email='joseph.pena@oebrasil.com.br',
    packages=find_packages(),
    install_requires=[
    ],

    classifiers=[
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3.9',
    ],
)
