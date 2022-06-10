from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from setuptools import find_packages
from setuptools import setup

description="Use GAN to generate particle physics events"

setup(
    name="gan4hep",
    version="0.4.0",
    description=description,
    long_description=description,
    author="Xiangyang Ju",
    license="Apache License, Version 2.0",
    keywords=["GAN", "HEP", "GNN"],
    url="https://github.com/xju2/gan4hep",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'tensorflow',
        'tensorflow-probability',
        "graph_nets@ https://github.com/deepmind/graph_nets/tarball/master",
        "matplotlib",
        'tqdm',
        'more_itertools',
        'scipy',
        'scikit-learn',
        'pyyaml>=5.1',
        'pandas',
    ],
    setup_requires=[],
    classifiers=[
        "Programming Language :: Python :: 3.8",
    ],
    scripts=[
        'gan4hep/gan/train_gan.py',
        'gan4hep/nf/train_nf.py',
    ],
)
