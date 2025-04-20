#!/usr/bin/env python
# coding: utf-8

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = fh.read().splitlines()
    # Remove comments and empty lines
    requirements = [line for line in requirements if not line.startswith('#') and line.strip() != '']

setup(
    name="CNN-n-GRU",
    version="1.0.0",
    author="Alaa Nfissi",
    author_email="alaa.nfissi@mail.concordia.ca",
    description="CNN-n-GRU: A deep learning model for speech emotion recognition",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/alaaNfissi/CNN-n-GRU-for-speech-emotion-recognition",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Sound/Audio :: Analysis",
    ],
    python_requires=">=3.6",
    install_requires=requirements,
    include_package_data=True,
) 