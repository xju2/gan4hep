#!/user/bin/env python

import os
import numpy as np
import tensorflow as tf

from gan import GAN
from aae import AdversarialAutoencoder
from cgan import CGAN
from wgan import WGAN

all_gans = ['GAN', "AdversarialAutoencoder", 'CGAN', 'WGAN']

from gan4hep.utils_gan import generate_and_save_images
from gan4hep.preprocess import herwig_angles

def inference(gan, test_in, test_truth, log_dir):
    checkpoint_dir = os.path.join(log_dir, "checkpoints")
    checkpoint = tf.train.Checkpoint(
        generator=gan.generator,
        discriminator=gan.discriminator)
    ckpt_manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=None)
    logging.info("Loading latest checkpoint from: {}".format(checkpoint_dir))
    _ = checkpoint.restore(ckpt_manager.latest_checkpoint).expect_partial()

    AUTO = tf.data.experimental.AUTOTUNE
    noise = np.random.normal(loc=0., scale=1., size=(test_truth.shape[0], self.noise_dim))
    test_in = np.concatenate(
        [test_in, noise], axis=1).astype(np.float32) if test_in is not None else noise
    testing_data = tf.data.Dataset.from_tensor_slices(
        (test_in, test_truth)).batch(batch_size, drop_remainder=True).prefetch(AUTO)

    summary_dir = os.path.join(log_dir, "logs_inference")
    summary_writer = tf.summary.create_file_writer(summary_dir)

    img_dir = os.path.join(log_dir, 'img_inference')
    os.makedirs(img_dir, exist_ok=True)
    tot_wdis = generate_and_save_images(
        self.generator, -1, testing_data, summary_writer, img_dir)
    print(tot_wdis)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Train The GAN')
    add_arg = parser.add_argument
    add_arg("model", choices=all_gans, help='gan model')
    add_arg("filename", help='input filename', default=None, nargs='+')
    add_arg("--epochs", help='number of maximum epochs', default=100, type=int)
    add_arg("--log-dir", help='log directory', default='log_training')
    add_arg("--num-test-evts", help='number of testing events', default=10000, type=int)
    add_arg("--inference", help='perform inference only', action='store_true')
    add_arg("-v", '--verbose', help='tf logging verbosity', default='INFO',
        choices=['WARN', 'INFO', "ERROR", "FATAL", 'DEBUG'])
    add_arg("--max-evts", help='Maximum number of events', type=int, default=None)
    add_arg("--batch-size", help='Batch size', type=int, default=512)
    add_arg("--lr", help='learning rate', type=float, default=0.0001)
    args = parser.parse_args()

    from tensorflow.compat.v1 import logging
    logging.set_verbosity(args.verbose)

    train_in, train_truth, test_in, test_truth = herwig_angles(
        args.filename, max_evts=args.max_evts)

    batch_size = args.batch_size
    gan = eval(args.model)()
    if args.inference:
        inference(gan, test_in, test_truth, args.log_dir)
    else:
        gan.train(
            train_truth, args.epochs, batch_size,
            test_truth, args.log_dir,
            generate_and_save_images,
            train_in, test_in)