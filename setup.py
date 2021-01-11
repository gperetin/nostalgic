#!/usr/bin/env python

from distutils.core import setup

setup(
    name='nostalgic',
    version='0.1dev',
    description='Backtesting system',
    author='Goran Peretin',
    author_email='goran.peretin@gmail.com',
    url='https://github.com/gperetin/nostalgic',
    packages=['nostalgic',],
    install_requires=[
        'pandas',
    ]
)
