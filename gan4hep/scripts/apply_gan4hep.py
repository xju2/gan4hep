#!/usr/bin/env python

from gan4hep import utils_gan
from gan4hep.utils_plot import load_yaml


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Apply The GAN')
    add_arg = parser.add_argument
    add_arg("filename", help='configuration name')

    args = parser.parse_args()

    config = load_yaml(args.filename)
    gan = utils_gan.load_model_from_ckpt(**config)