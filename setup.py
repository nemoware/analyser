#!/usr/bin/python
# -*- coding: utf-8 -*-
# coding=utf-8
from pip._internal.req import parse_requirements
from setuptools import setup, find_packages, find_namespace_packages




with open("README.md", "r") as fh:
  long_description = fh.read()


# parse_requirements() returns generator of pip.req.InstallRequirement objects
install_reqs = parse_requirements('requirements.txt', session='hack')

# reqs is a list of requirement
reqs = [str(ir.req) for ir in install_reqs]

setup(
  name="nemoware-analyzer",
  version="0.0.1",

  description="GPN Audit: NLP analyser",
  long_description=long_description,
  long_description_content_type="text/markdown",
  url="https://github.com/nemoware/analyser",


  install_requires=reqs,
  packages=find_namespace_packages(exclude=["tests", "notebooks"]),
  classifiers=[
    "Programming Language :: Python :: 3",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
  ],
  python_requires='>=3.6',



  entry_points={"console_scripts": ["analyser-run = bin.analyser_run:main"]},
  scripts=["bin/analyser-run.cmd"],
)