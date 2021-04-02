#!/usr/bin/env python
"""
Training a GAN for modeling hadronic interactions
"""
# import tensorflow.experimental.numpy as tnp

import os
import sys
import argparse
import importlib

import re
import time
import random
import functools
import six
from types import SimpleNamespace

from scipy import stats
import numpy as np
import tqdm

import gan4hep
from gan4hep.gan_base import GANOptimizer

from gan4hep.graph import loop_dataset
from gan4hep.graph import read_dataset

import tensorflow as tf
from tensorflow.compat.v1 import logging
logging.info("TF Version:{}".format(tf.__version__))

node_mean = np.array([
    [14.13, 0.05, -0.10, -0.04], 
    [7.73, 0.02, -0.04, -0.08],
    [6.41, 0.04, -0.06, 0.04]
], dtype=np.float32)

node_scales = np.array([
    [13.29, 10.54, 10.57, 12.20], 
    [8.62, 6.29, 6.35, 7.29],
    [6.87, 5.12, 5.13, 5.90]
], dtype=np.float32)


node_abs_max = np.array([
    [49.1, 47.7, 46.0, 47.0],
    [46.2, 40.5, 41.0, 39.5],
    [42.8, 36.4, 37.0, 35.5]
], dtype=np.float32)

max_energy_px_py_pz = np.array([49.1, 47.7, 46.0, 47.0], dtype=np.float32)
max_pt_eta_phi_energy = np.array([50, 5, np.pi, 50], dtype=np.float32)


gan_types = ['mlp_gan', 'rnn_mlp_gan', 'rnn_rnn_gan']
def create_gan(gan_name, ls=512, nl=10):
    gan_module = importlib.import_module("gan4hep."+gan_name)
    return gan_module.GAN(latent_size=ls, num_layers=nl)


def init_workers(distributed=False):
    return SimpleNamespace(rank=0, size=1, local_rank=0, local_size=1, comm=None)


def get_pt_eta_phi(px, py, pz):
    p = np.sqrt(px**2 + py**2 + pz**2)
    pt = np.sqrt(px**2 + py**2)
    phi = np.arctan2(py, px)
    theta = np.arccos(pz/p)
    eta = -np.log(np.tan(0.5*theta))
    return pt,eta,phi


def train_and_evaluate(args):
    dist = init_workers(args.distributed)
    batch_size = args.batch_size

    device = 'CPU'
    gpus = tf.config.experimental.list_physical_devices("GPU")
    logging.info("found {} GPUs".format(len(gpus)))

    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    if len(gpus) > 0:
        device = "{}GPUs".format(len(gpus))
    if gpus and args.distributed:
        tf.config.experimental.set_visible_devices(
            gpus[hvd.local_rank()], 'GPU')

    time_stamp = time.strftime('%Y%m%d-%H%M%S', time.localtime())
    output_dir = args.output_dir
    # output_dir = os.path.join(args.output_dir, "{}".format(time_stamp))
    if dist.rank == 0:
        os.makedirs(output_dir, exist_ok=True)
    logging.info("Checkpoints and models saved at {}".format(output_dir))

    n_epochs = args.max_epochs
    logging.info("{} epochs with batch size {}".format(
        n_epochs, batch_size))
    logging.info("I am in hvd rank: {} of  total {} ranks".format(
        dist.rank, dist.size))

    if dist.rank == 0:
        train_input_dir = os.path.join(args.input_dir, 'train')
        val_input_dir = os.path.join(args.input_dir, 'val')
        train_files = tf.io.gfile.glob(
            os.path.join(train_input_dir, args.patterns))
        eval_files = tf.io.gfile.glob(
            os.path.join(val_input_dir, args.patterns))
        # split the number of files evenly to all ranks
        train_files = [x.tolist()
                       for x in np.array_split(train_files, dist.size)]
        eval_files = [x.tolist()
                      for x in np.array_split(eval_files, dist.size)]
    else:
        train_files = None
        eval_files = None

    if args.distributed:
        train_files = dist.comm.scatter(train_files, root=0)
        eval_files = dist.comm.scatter(eval_files, root=0)
    else:
        train_files = train_files[0]
        eval_files = eval_files[0]

    if args.debug:
        train_files = train_files[0:1]
        eval_files = eval_files[0:1]

    logging.info("rank {} has {} training files and {} evaluation files".format(
        dist.rank, len(train_files), len(eval_files)))
    input_frac = args.input_frac
    if input_frac < 1 and input_frac > 0:
        n_tr = int(len(train_files)*input_frac) + 1
        n_ev = int(len(eval_files)* input_frac) + 1
        train_files, eval_files = train_files[:n_tr], eval_files[:n_ev]
        logging.info("However, only {} fraction of inputs were reqested".format(input_frac))
        logging.info("rank {} has {} training files and {} evaluation files".format(
            dist.rank, len(train_files), len(eval_files)))


    AUTO = tf.data.experimental.AUTOTUNE
    training_dataset, ngraphs_train = read_dataset(train_files, args.evts_per_file)
    training_dataset = training_dataset.repeat(n_epochs+1).prefetch(AUTO)
    if args.shuffle_size > 0:
        training_dataset = training_dataset.shuffle(
                args.shuffle_size, seed=12345, reshuffle_each_iteration=False)

    validating_dataset, ngraphs_val = read_dataset(eval_files, args.evts_per_file)
    validating_dataset = validating_dataset.repeat().prefetch(AUTO)


    logging.info("rank {} has {:,} training events and {:,} validating events".format(
        dist.rank, ngraphs_train, ngraphs_val))

    gan = create_gan(args.gan_type)

    optimizer = GANOptimizer(
                        gan,
                        batch_size=batch_size,
                        noise_dim=args.noise_dim,
                        num_epcohs=n_epochs,
                        disc_lr=args.disc_lr,
                        gen_lr=args.gen_lr,
                        with_disc_reg=args.with_disc_reg, 
                        gamma_reg=args.gamma_reg
                        )
    
    disc_step = optimizer.disc_step
    step = optimizer.step
    if not args.debug:
        step = tf.function(step)
        disc_step = tf.function(disc_step)


    training_data = loop_dataset(training_dataset, batch_size)
    validating_data = loop_dataset(validating_dataset, batch_size)
    steps_per_epoch = ngraphs_train // batch_size


    log_dir = os.path.join(output_dir, "logs/{}/train".format(time_stamp))
    train_summary_writer = tf.summary.create_file_writer(log_dir)

    ckpt_dir = os.path.join(output_dir, "checkpoints")
    checkpoint = tf.train.Checkpoint(
        optimizer=optimizer,
        gan=gan)
    ckpt_manager = tf.train.CheckpointManager(checkpoint, directory=ckpt_dir,
                                              max_to_keep=5, keep_checkpoint_every_n_hours=8)
    logging.info("Loading latest checkpoint from: {}".format(ckpt_dir))
    _ = checkpoint.restore(ckpt_manager.latest_checkpoint)

    
    def normalize(inputs, targets):
        # node features are [energy, px, py, pz]
        if args.use_pt_eta_phi_e:
            # inputs
            pt, eta, phi = get_pt_eta_phi(
                inputs.nodes[:, 1], inputs.nodes[:, 2], inputs.nodes[:, 3])
            input_nodes = np.stack([pt, eta, phi, inputs.nodes[:, 0]], axis=1) / max_pt_eta_phi_energy
            # outputs
            o_pt, o_eta, o_phi = get_pt_eta_phi(
                targets.nodes[:, 1], targets.nodes[:, 2], targets.nodes[:, 3])
            target_nodes = np.stack([o_pt, o_eta, o_phi, targets.nodes[:, 0]], axis=1) / max_pt_eta_phi_energy
        else:            
            input_nodes = (inputs.nodes - node_mean[0])/node_scales[0]
            target_nodes = targets.nodes / max_energy_px_py_pz
        target_nodes = np.reshape(target_nodes, [batch_size, -1])
        return input_nodes, target_nodes

    if args.warm_up:
        # train discriminator for certain batches
        # to "warm up" the discriminator
        print("start to warm up discriminator with {} batches".format(args.disc_batches))
        for _ in range(args.disc_batches):
            inputs_tr, targets_tr = next(training_data)
            input_nodes, target_nodes = normalize(inputs_tr, targets_tr)
            
            input_nodes = tf.convert_to_tensor(input_nodes, dtype=tf.float32)
            target_nodes = tf.convert_to_tensor(target_nodes, dtype=tf.float32)
            disc_step(target_nodes, input_nodes)

        print("finished the warm up")

    pre_gen_loss = pre_disc_loss = 0
    start_time = time.time()

    with tqdm.trange(n_epochs*steps_per_epoch) as t:

        for step_num in t:
            epoch = tf.constant(int(step_num / steps_per_epoch), dtype=tf.int32)
            inputs_tr, targets_tr = next(training_data)

            # --------------------------------------------------------
            # scale the inputs and outputs to [-1, 1]
            input_nodes, target_nodes = normalize(inputs_tr, targets_tr)
            input_nodes = tf.convert_to_tensor(input_nodes, dtype=tf.float32)
            target_nodes = tf.convert_to_tensor(target_nodes, dtype=tf.float32)
            # --------------------------------------------------------

            disc_loss, gen_loss, lr_mult = step(target_nodes, epoch, input_nodes)
            if args.with_disc_reg:
                disc_loss, disc_reg, grad_D_true_logits_norm, grad_D_gen_logits_norm, reg_true, reg_gen = disc_loss
            else:
                disc_loss, = disc_loss

            if step_num == 0:
                print(">>>{:,} trainable variables in Generator; {:,} trainable variables in Discriminator<<<".format(
                    *optimizer.gan.num_trainable_vars()
                ))

            disc_loss = disc_loss.numpy()
            gen_loss = gen_loss.numpy()
            if step_num and (step_num % args.log_freq == 0):            
                ckpt_manager.save()
                pre_gen_loss = gen_loss
                pre_disc_loss = disc_loss

                # adding testing results
                predict_4vec = []
                truth_4vec = []
                for ib in range(args.val_batches):
                    inputs_val, targets_val = normalize(* (next(validating_data)))
                    gen_evts_val = optimizer.cond_gen(inputs_val)
                    predict_4vec.append(tf.reshape(gen_evts_val, [batch_size, -1, 4]))
                    truth_4vec.append(tf.reshape(targets_val, [batch_size, -1, 4])[:, 1:, :])
        
                predict_4vec = tf.concat(predict_4vec, axis=0)
                truth_4vec = tf.concat(truth_4vec, axis=0)

                # log some metrics
                this_epoch = time.time()
                with train_summary_writer.as_default():
                    tf.summary.experimental.set_step(step_num)
                    # epoch = epoch.numpy()
                    tf.summary.scalar("gen_loss", gen_loss, description='generator loss')
                    tf.summary.scalar("discr_loss", disc_loss, description="discriminator loss")
                    tf.summary.scalar("time", (this_epoch-start_time)/60.)
                    if args.with_disc_reg:
                        tf.summary.scalar("discr_reg", disc_reg.numpy().mean(), description='discriminator regularization')
                        tf.summary.scalar("reg_gen", reg_gen.numpy().mean(), description='regularization on generated events')
                        tf.summary.scalar("reg_true", reg_true.numpy().mean(), description='regularization on truth events')
                        tf.summary.scalar("grad_D1_logits_norm", grad_D_true_logits_norm.numpy().mean(),
                                    description="gradients of true logits")
                        tf.summary.scalar("grad_D2_logits_norm", grad_D_gen_logits_norm.numpy().mean(),
                                    description="gradients of generated logits")
                    
                    # plot the eight variables and resepctive Wasserstein distance (i.e. Earch Mover Distance)
                    # Use the Kolmogorov-Smirnov test, 
                    # it turns a two-sided test for the null hypothesis that the two distributions
                    # are drawn from the same continuous distribution.
                    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.combine_pvalues.html#scipy.stats.combine_pvalues
                    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ks_2samp.html#scipy.stats.ks_2samp
                    # https://docs.scipy.org/doc/scipy/reference/stats.html#statistical-distances
                    predict_4vec = tf.reshape(predict_4vec, [batch_size, -1]).numpy()
                    truth_4vec = tf.reshape(truth_4vec, [batch_size, -1]).numpy()
                    distances = []

                    for icol in range(predict_4vec.shape[1]):
                        dis = stats.wasserstein_distance(predict_4vec[:, icol], truth_4vec[:, icol])
                        _, pvalue = stats.ks_2samp(predict_4vec[:, icol], truth_4vec[:, icol])
                        if pvalue < 1e-6: pvalue = 1e-6
                        energy_dis = stats.energy_distance(predict_4vec[:, icol], truth_4vec[:, icol])

                        tf.summary.scalar("wasserstein_distance_var{}".format(icol), dis)
                        tf.summary.scalar("energy_distance_var{}".format(icol), energy_dis)
                        tf.summary.scalar("KS_Test_var{}".format(icol), pvalue)
                        distances.append([dis, energy_dis, pvalue])
                    distances = np.array(distances, dtype=np.float32)
                    tot_wdis = sum(distances[:, 0])
                    tot_edis = sum(distances[:, 1])
                    tf.summary.scalar("tot_wasserstein_dis", tot_wdis, description="total wasserstein distance")
                    tf.summary.scalar("tot_energy_dis", tot_edis, description="total energy distance")
                    _ , comb_pvals = stats.combine_pvalues(distances[:, 2], method='fisher')
                    tf.summary.scalar("tot_KS", comb_pvals, description="total wasserstein distance")

                t.set_description('Epoch {}/{}'.format(epoch.numpy(), n_epochs))
                t.set_postfix(
                    G_loss=gen_loss, D_loss=disc_loss,
                    Wdis=tot_wdis, Pval=comb_pvals, Edis=tot_edis,
                )   


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Train nx-graph with configurations')
    add_arg = parser.add_argument
    add_arg("input_dir",
            help='input directory that contains subfolder of train, val and test')
    add_arg("output_dir", help="where the model and training info saved")
    add_arg("--input-frac", help="use a fraction of input files", default=1., type=float)
    add_arg("--use-pt-eta-phi-e", help='use [pT, eta, phi, E]', action='store_true')
    add_arg("--gan-type", help='which gan to use', required=True, choices=gan_types)
    add_arg("--patterns", help='file patterns', default='*')
    add_arg('-d', '--distributed', action='store_true',
            help='data distributed training')
    add_arg("--disc-lr", help='learning rate for discriminator', default=2e-4, type=float)
    add_arg("--gen-lr", help='learning rate for generator', default=5e-5, type=float)
    add_arg("--max-epochs", help='number of epochs', default=1, type=int)
    add_arg("--batch-size", type=int,
            help='training/evaluation batch size', default=500)
    add_arg("--shuffle-size", type=int,
            help="number of events for shuffling", default=650)
    add_arg("--warm-up", action='store_true', help='warm up discriminator first')
    add_arg("--disc-batches", help='number of batches training discriminator only', type=int, default=100)
    add_arg("--loss-type", choices=['logloss', 'mse'], default='logloss')

    add_arg("--noise-dim", type=int, help='dimension of noises', default=8)
    add_arg("--disc-num-iters", type=int,
            help='number of message passing for discriminator', default=4)
    add_arg("--disc-alpha", type=float,
            help='inversely scale true dataset in the loss calculation', default=0.1)
    add_arg("--disc-beta", type=float,
            help='scales generated dataset in the loss calculation', default=0.8)
    add_arg("--with-disc-reg", action='store_true', help='with discriminator regularizer')
    add_arg("--gamma-reg", type=float, help="scale the regularization term", default=1e-3)
    add_arg("--log-freq", type=int, help='log per number of steps', default=50)
    add_arg("--val-batches", type=int, default=1, help='number of batches for validation')
    add_arg("--evts-per-file", default=5000, type=int, help='number of events per input file')

    add_arg("-v", "--verbose", help='verbosity', choices=['DEBUG', 'ERROR', 'FATAL', 'INFO', 'WARN'],
            default="INFO")
    add_arg("--debug", help='in debug mode', action='store_true')
    # args, _ = parser.parse_known_args()
    args = parser.parse_args()
    # print(args)

    # Set python level verbosity
    logging.set_verbosity(args.verbose)
    # Suppress C++ level warnings.
    # os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    train_and_evaluate(args)