from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from setuptools import find_packages
from setuptools import setup

description="Use GAN to generate particle physics events"

setup(
    name="gan4hep",
    version="0.1.0",
    description=description,
    long_description=description,
    author="Xiangyang Ju",
    license="Apache License, Version 2.0",
    keywords=["GAN", "HEP"],
    url="https://github.com/xju2/gan4hep",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'tensorflow >= 2.4.0',
        "graph_nets@ https://github.com/deepmind/graph_nets/tarball/master",
        "matplotlib",
        'tqdm',
        'more_itertools',
        'scipy',
    ],
    # package_data = {
    #     "gan4hep": ["config/*.yaml"]
    # },
    setup_requires=[],
    classifiers=[
        "Programming Language :: Python :: 3.8",
    ],
    scripts=[
        'gan4hep/scripts/train_gan4hep.py',
    ],
)