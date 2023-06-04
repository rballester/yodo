#!/usr/bin/env python
# -*- coding: utf-8 -*-


try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


with open('README.md') as readme_file:
    readme = readme_file.read()

with open('HISTORY.md') as history_file:
    history = history_file.read()

setup(
    name='yodo',
    version='0.1.0',
    description="Variance-based Sensitivity Analysis of Bayesian Networks",
    long_description="You Only Derive Once (YODO) computes all sensitivity values of a Bayesian Network, assuming proportional covariation. See https://proceedings.mlr.press/v186/ballester-ripoll22a.html",
    url='https://github.com/rballester/yodo',
    author="Rafael Ballester-Ripoll",
    author_email='rafael.ballester@ie.edu',
    packages=[
        'yodo',
    ],
    include_package_data=True,
    install_requires=[
        'pgmpy',
        'numpy',
        'torch',
    ],
    license="BSD",
    zip_safe=False,
    keywords='yodo',
    classifiers=[
        'License :: OSI Approved',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
    ],
    test_suite='tests',
    tests_require='pytest'
)
