#!/user/bin/env python

import os
import numpy as np
import tensorflow as tf

from gan import GAN
from aae import AAE
from cgan import CGAN
from wgan import WGAN

all_gans = ['GAN', "AAE", 'CGAN', 'WGAN']

from gan4hep.utils_gan import generate_and_save_images
from gan4hep.utils_gan import generate_and_save_images_end_of_run
from gan4hep.preprocess import herwig_angles
from gan4hep.preprocess import dimuon_inclusive

def inference(gan, test_in, test_truth, log_dir): 
    checkpoint_dir = os.path.join(log_dir, "checkpoints")
    checkpoint = tf.train.Checkpoint(
        generator=gan.generator,
        discriminator=gan.discriminator)
    ckpt_manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=None)
    logging.info("Loading latest checkpoint from: {}".format(checkpoint_dir))
    _ = checkpoint.restore(ckpt_manager.latest_checkpoint).expect_partial()

    AUTO = tf.data.experimental.AUTOTUNE
    noise = np.random.normal(loc=0., scale=1., size=(test_truth.shape[0], gan.noise_dim))
    test_in = np.concatenate(
        [test_in, noise], axis=1).astype(np.float32) if test_in is not None else noise
    testing_data = tf.data.Dataset.from_tensor_slices(
        (test_in, test_truth)).batch(batch_size, drop_remainder=True).prefetch(AUTO)

    summary_dir = os.path.join(log_dir, "logs_inference")
    summary_writer = tf.summary.create_file_writer(summary_dir)

    img_dir = os.path.join(log_dir, 'img_inference')
    os.makedirs(img_dir, exist_ok=True)
    tot_wdis = generate_and_save_images(
        gan.generator, -1, testing_data, summary_writer, img_dir)
    print(tot_wdis)


if __name__ == '__main__': #Code only runs if its called by the terminal and not by another .py file
    
    
    # How to alter the parameters of the GAN in the CMD line terminal
    
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
    add_arg("--max-evts", help='Maximum number of events', type=int, default=10000)
    add_arg("--batch-size", help='Batch size', type=int, default=512)
    add_arg("--lr-dis", help='learning rate discriminator', type=float, default=0.0001)
    add_arg("--lr-gen", help='learning rate generator', type=float, default=0.0001)
    add_arg("--test-frac", help='Fraction of data used for testing', type=float, default=0.1)
    add_arg("--single-step-limit", help='Number of epochs before training multiple times per epoch', type=int, default=5)
    add_arg("--data", default='herwig_angles',
        choices=['herwig_angles', 'dimuon_inclusive'])

    # model parameters
    add_arg("--gen-layers", type=int, default=0, help='Number of extra layers to add to generator model with 256 nodes')
    add_arg("--dis-layers", type=int, default=0, help='Number of extra layers to add to discriminator model with 256 nodes')
    add_arg("--gen-train-num", type=int, default=1, help='Number of extra optimizing attemps for each epoch for the generator')
    add_arg("--dis-train-num", type=int, default=1,help='Number of extra optimizing attemps for each epoch for the discriminator')
    add_arg("--noise-dim", type=int, default=4, help="noise dimension")
    add_arg("--num-nodes", type=int, default=256, help="Number of Nodes in a NN Layer")
    add_arg("--gen-output-dim", type=int, default=2, help='generator output dimension')
    add_arg("--cond-dim", type=int, default=0, help='dimension of conditional input')
    add_arg("--disable-tqdm", action="store_true", help='disable tqdm')
    add_arg("--noise-type", default='gaussian',
            choices=['gaussian','uniform'])
    
    #Creates namespace
   
    args = parser.parse_args()

    from tensorflow.compat.v1 import logging
    logging.set_verbosity(args.verbose) 

    # prepare input data by calling those function implemented in 
    # gan4hep.preprocess.
    train_in, train_truth, test_in, test_truth = eval(args.data)(
        args.filename, max_evts=args.max_evts,testing_frac=args.test_frac) #Split into train and test dataset with max number of events and finds out whether args.data is set to herwig_angles or dimuon_inclusive

    batch_size = args.batch_size # Get input batch size
    
    from tensorflow import keras
    from tensorflow.keras import layers
    
    gan = eval(args.model)(**vars(args)) # eval(args.model) FInds what type of NN has been picked in the input and find the           corresponding class
    
    #Define variables from args to be given to gan.train (These lines might be useless)
    gan.lr=args.lr_dis
    lr_dis=args.lr_dis
    lr_gen=args.lr_gen
    gen_layers=args.gen_layers
    dis_layers=args.dis_layers
    noise_type=args.noise_type
    num_nodes=args.num_nodes
    gen_train_num=args.gen_train_num
    dis_train_num=args.dis_train_num
    single_step_limit=args.single_step_limit


    if args.inference:
        inference(gan, test_in, test_truth, args.log_dir) # Run inference function at the top of the page
    else:
        gan.train(args,
            train_truth, args.epochs, batch_size,
            test_truth, args.log_dir,
       generate_and_save_images,generate_and_save_images_end_of_run,lr_dis,lr_gen,noise_type,gen_layers,dis_layers,num_nodes,gen_train_num,dis_train_num,single_step_limit, train_in, test_in)
