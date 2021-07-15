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
import sklearn.metrics
import numpy as np
import tqdm

import gan4hep
from gan4hep.gan_base import GANOptimizer
from gan4hep import data_handler as DataHandler
from gan4hep.graph import loop_dataset
from gan4hep.graph import read_dataset
from gan4hep import utils_gan as GanUtils

import tensorflow as tf
from tensorflow.compat.v1 import logging #
logging.info("TF Version:{}".format(tf.__version__))


gan_types = ['mlp_gan', 'rnn_mlp_gan', 'rnn_rnn_gan', 'gnn_gnn_gan']

def init_workers(distributed=False):
    return SimpleNamespace(rank=0, size=1, local_rank=0, local_size=1, comm=None)

def get_pt_eta_phi(px, py, pz):
    p = np.sqrt(px**2 + py**2 + pz**2)
    pt = np.sqrt(px**2 + py**2)
    phi = np.arctan2(py, px)
    theta = np.arccos(pz/p)
    eta = -np.log(np.tan(0.5*theta))
    return pt,eta,phi

def train_and_evaluate(
    batch_size, # batch size [HPO]
    max_epochs, # maximum epochs 
    noise_dim,  # noise dimension [HPO]
    disc_lr,    # discriminator learning rate [HPO]
    gen_lr,     # generator learning rate [HPO]
    gamma_reg,  # strength of regularization term [HPO]
    input_dir,  # input directory
    output_dir, # output directory
    layer_size=512, # layer size in MLP [HPO]
    num_layers=10,  # number of layers [HPO]
    patterns="*",
    gan_type='rnn_mlp_gan',
    with_disc_reg=True,
    distributed=False,
    evts_per_file=5000,
    shuffle_size=-1,
    debug=False,
    input_frac=0.1, # to use a fraction of training events
    do_log=False,
    warm_up=True,
    disc_batches=10, # number of batches for warming up discriminator
    val_batches=2, # number of batches for validation
    log_freq=1000, # number of batches per log
    use_pt_eta_phi_e=False, # Keep it false, not working...use [pt, eta, phi, E] as inputs, possible HPO
    decay_epochs=2,
    decay_base=0.96,
    disable_tqdm=False,
    hadronic=False, ## if data is hadronic interactions
    num_processing_steps=None,
    *args, **kwargs
):
    """
    The reason that hyperparameters are given as parameters is that
    one can easily perform HPO.
    """
    dist = init_workers(distributed)

    device = 'CPU'
    gpus = tf.config.experimental.list_physical_devices("GPU")
    logging.info("found {} GPUs".format(len(gpus)))

    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    if len(gpus) > 0:
        device = "{}GPUs".format(len(gpus))
    if gpus and distributed:
        tf.config.experimental.set_visible_devices(
            gpus[dist.local_rank()], 'GPU')

    time_stamp = time.strftime('%Y%m%d-%H%M%S', time.localtime())
    if dist.rank == 0:
        os.makedirs(output_dir, exist_ok=True)
    logging.info("Checkpoints and models saved at {}".format(output_dir))

    logging.info("{} epochs with batch size {}".format(
        max_epochs, batch_size))
    logging.info("I am in hvd rank: {} of  total {} ranks".format(
        dist.rank, dist.size))

    if dist.rank == 0:
        train_input_dir = os.path.join(input_dir, 'train')
        val_input_dir = os.path.join(input_dir, 'val')
        train_files = tf.io.gfile.glob(
            os.path.join(train_input_dir, patterns))
        eval_files = tf.io.gfile.glob(
            os.path.join(val_input_dir, patterns))
        # split the number of files evenly to all ranks
        train_files = [x.tolist()
                       for x in np.array_split(train_files, dist.size)]
        eval_files = [x.tolist()
                      for x in np.array_split(eval_files, dist.size)]
    else:
        train_files = None
        eval_files = None

    if distributed:
        train_files = dist.comm.scatter(train_files, root=0)
        eval_files = dist.comm.scatter(eval_files, root=0)
    else:
        train_files = train_files[0]
        eval_files = eval_files[0]

    if debug:
        train_files = train_files[0:1]
        eval_files = eval_files[0:1]

    logging.info("rank {} has {} training files and {} evaluation files".format(
        dist.rank, len(train_files), len(eval_files)))
    if input_frac < 1 and input_frac > 0:
        n_tr = int(len(train_files)*input_frac) + 1
        n_ev = int(len(eval_files)* input_frac) + 1
        train_files, eval_files = train_files[:n_tr], eval_files[:n_ev]
        logging.info("However, only {} fraction of inputs were reqested".format(input_frac))
        logging.info("rank {} has {} training files and {} evaluation files".format(
            dist.rank, len(train_files), len(eval_files)))


    AUTO = tf.data.experimental.AUTOTUNE
    training_dataset, ngraphs_train = read_dataset(train_files, evts_per_file)
    training_dataset = training_dataset.repeat().prefetch(AUTO)
    if shuffle_size > 0:
        training_dataset = training_dataset.shuffle(
                shuffle_size, seed=12345, reshuffle_each_iteration=False)

    validating_dataset, ngraphs_val = read_dataset(eval_files, evts_per_file)
    validating_dataset = validating_dataset.repeat().prefetch(AUTO)


    logging.info("rank {} has {:,} training events and {:,} validating events".format(
        dist.rank, ngraphs_train, ngraphs_val))

    gan = GanUtils.create_gan(
        gan_type, noise_dim, batch_size,
        layer_size, num_layers, num_processing_steps)

    optimizer = GanUtils.create_optimizer(
                        gan, max_epochs,
                        disc_lr, gen_lr,
                        with_disc_reg, gamma_reg,
                        decay_epochs, decay_base, debug)
    
    disc_step = optimizer.disc_step
    step = optimizer.step
    if not debug:
        step = tf.function(step)
        disc_step = tf.function(disc_step)

    training_data = loop_dataset(training_dataset, batch_size)
    validating_data = loop_dataset(validating_dataset, batch_size)
    steps_per_epoch = ngraphs_train // batch_size

    log_dir = os.path.join(output_dir, "logs/{}/train".format(time_stamp))
    train_summary_writer = tf.summary.create_file_writer(log_dir)

    ckpt_dir = GanUtils.get_ckptdir(output_dir)
    checkpoint = tf.train.Checkpoint(
        optimizer=optimizer,
        gan=gan)
    step_counter = tf.Variable(0, trainable=False, dtype=tf.int32, name='step_counter')
    ckpt_manager = tf.train.CheckpointManager(checkpoint, directory=ckpt_dir,
                                              max_to_keep=10, keep_checkpoint_every_n_hours=1,
                                              step_counter=step_counter
                                            )
    logging.info("Loading latest checkpoint from: {}".format(ckpt_dir))
    _ = checkpoint.restore(ckpt_manager.latest_checkpoint)


    if warm_up:
        # train discriminator for certain batches
        # to "warm up" the discriminator
        print("start to warm up discriminator with {} batches".format(disc_batches))
        for _ in range(disc_batches):
            inputs_tr, targets_tr = next(training_data)

            input_nodes, target_nodes = DataHandler.normalize(
                inputs_tr, targets_tr, batch_size, hadronic=hadronic)
            disc_step(target_nodes, input_nodes)

        print("finished the warm up")

    start_time = time.time()

    wdis_all = []
    min_wdis = 9999
    with tqdm.trange(max_epochs, disable=disable_tqdm) as t0:
        for epoch in t0:
            t0.set_description('Epoch {}/{}'.format(epoch, max_epochs))
            with tqdm.trange(steps_per_epoch, disable=disable_tqdm) as t:
                for step_num in t:
                    # --------------------------------------------------------
                    input_nodes, target_nodes = DataHandler.normalize(* next(training_data), batch_size=batch_size, hadronic=hadronic)
                    # --------------------------------------------------------

                    disc_loss, gen_loss, lr_mult = step(target_nodes, epoch, input_nodes)
                    if with_disc_reg:
                        disc_loss, disc_reg, grad_D_true_logits_norm, grad_D_gen_logits_norm, reg_true, reg_gen = disc_loss
                    else:
                        disc_loss = disc_loss[0]

                    if step_num==0 and epoch == 0:
                        print(">>>{:,} trainable variables in Generator; "
                            "{:,} trainable variables in Discriminator<<<".format(
                            *optimizer.gan.num_trainable_vars()
                        ))

                    disc_loss = disc_loss.numpy()
                    gen_loss = gen_loss.numpy()
                    if step_num and (step_num % log_freq == 0):
                        tot_steps = epoch*steps_per_epoch + step_num
                        step_counter.assign(tot_steps)

                        # adding testing results
                        predict_4vec = []
                        truth_4vec = []
                        gen_scores = []
                        g4_scores = []
                        for _ in range(val_batches):
                            inputs_val, targets_val = DataHandler.normalize(* next(validating_data), batch_size=batch_size, hadronic=hadronic)
                            gen_evts_val = gan.generate(inputs_val, is_training=False)
                            predict_4vec.append(gen_evts_val)
                            truth_4vec.append(targets_val)
  
                            # check the performance of discriminator
                            gen_scores.append(tf.sigmoid(gan.discriminate(gen_evts_val, is_training=False)))
                            g4_scores.append(tf.sigmoid(gan.discriminate(targets_val, is_training=False)))
                
                        predict_4vec = tf.concat(predict_4vec, axis=0)
                        truth_4vec = tf.concat(truth_4vec, axis=0)


                        all_scores = tf.concat(gen_scores + g4_scores, axis=0)
                        truth_scores = tf.concat([tf.zeros_like(x) for x in gen_scores] \
                            + [tf.ones_like(x) for x in g4_scores], axis=0)
                        y_true = (truth_scores > 0.5)
                        fpr, tpr, _ = sklearn.metrics.roc_curve(y_true, all_scores)
                        disc_auc = sklearn.metrics.auc(fpr, tpr)

                        # log some metrics
                        this_epoch = time.time()
                        with train_summary_writer.as_default():
                            tf.summary.experimental.set_step(tot_steps)
                            # epoch = epoch.numpy()
                            tf.summary.scalar("gen_loss", gen_loss, description='generator loss')
                            tf.summary.scalar("discr_loss", disc_loss, description="discriminator loss")
                            tf.summary.scalar("disc_auc", disc_auc, description="AUC of disciminator")
                            tf.summary.scalar("time", (this_epoch-start_time)/60.)
                            if with_disc_reg:
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

                            predict_4vec = predict_4vec.numpy()
                            truth_4vec = truth_4vec.numpy()
                            distances = []

                            for icol in range(4, predict_4vec.shape[1]):
                                # print("predict -->", icol, predict_4vec[:, icol])
                                # print("truth -->", icol, truth_4vec[:, icol])
                                dis = stats.wasserstein_distance(predict_4vec[:, icol], truth_4vec[:, icol])
                                _, pvalue = stats.ks_2samp(predict_4vec[:, icol], truth_4vec[:, icol])
                                if pvalue < 1e-6: pvalue = 1e-6
                                energy_dis = stats.energy_distance(predict_4vec[:, icol], truth_4vec[:, icol])
                                mse_loss = np.sum((predict_4vec[:, icol] - truth_4vec[:, icol])**2)/predict_4vec.shape[0]

                                tf.summary.scalar("wasserstein_distance_var{}".format(icol), dis)
                                tf.summary.scalar("energy_distance_var{}".format(icol), energy_dis)
                                tf.summary.scalar("KS_Test_var{}".format(icol), pvalue)
                                tf.summary.scalar("MSE_distance_var{}".format(icol), mse_loss)
                                distances.append([dis, energy_dis, pvalue, mse_loss])
                            distances = np.array(distances, dtype=np.float32)
                            tot_wdis = sum(distances[:, 0]) / distances.shape[0]
                            tot_edis = sum(distances[:, 1]) / distances.shape[0]
                            tf.summary.scalar("tot_wasserstein_dis", tot_wdis, description="total wasserstein distance")
                            tf.summary.scalar("tot_energy_dis", tot_edis, description="total energy distance")
                            _ , comb_pvals = stats.combine_pvalues(distances[:, 2], method='fisher')
                            tf.summary.scalar("tot_KS", comb_pvals, description="total KS pvalues distance")
                            tot_mse = sum(distances[:, 3]) / distances.shape[0]
                            tf.summary.scalar("tot_mse", tot_mse, description='mean squared loss')

                        t.set_postfix(
                            G_loss=gen_loss, D_loss=disc_loss, D_AUC=disc_auc,
                            Wdis=tot_wdis, Pval=comb_pvals, Edis=tot_edis, MSE=tot_mse)
                        wdis_all.append(tot_wdis)
                        # only save the model if the Wasserstein distance is smaller
                        if tot_wdis < min_wdis:
                            min_wdis = tot_wdis
                            ckpt_manager.save()
    return min_wdis


def add_training_args(parser):
    add_arg = parser.add_argument
    add_arg("input_dir",
            help='input directory that contains subfolder of train, val and test')
    add_arg("output_dir", help="where the model and training info saved")
    add_arg("--input-frac", help="use a fraction of input files", default=1., type=float)
    # add_arg("--use-pt-eta-phi-e", help='use [pT, eta, phi, E]', action='store_true')
    add_arg("--gan-type", help='which gan to use', required=True, choices=gan_types)
    add_arg("--layer-size", help='MLP layer size', default=512, type=int)
    add_arg("--num-layers", help='MLP number of layers', default=10, type=int)
    add_arg("--num-processing-steps", default=None, type=int, help="number of processing steps")
    add_arg("--patterns", help='file patterns', default='*')
    # add_arg('-d', '--distributed', action='store_true',
    #         help='data distributed training')
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
    add_arg("--with-disc-reg", action='store_true', help='with discriminator regularizer')

    # learning rate decay --> decay_base ^ (epoch / decay_epoch)
    add_arg("--decay-epochs", type=int, help='how often learning rate decays', default=2)
    add_arg("--decay-base", type=float, help='base value for learning rate decay', default=0.96)

    #
    add_arg("--gamma-reg", type=float, help="scale the regularization term", default=1e-3)
    add_arg("--log-freq", type=int, help='log per number of steps', default=50)
    add_arg("--val-batches", type=int, default=1, help='number of batches for validation')
    add_arg("--evts-per-file", default=5000, type=int, help='number of events per input file')

    add_arg("-v", "--verbose", help='verbosity', choices=['DEBUG', 'ERROR', 'FATAL', 'INFO', 'WARN'],
            default="INFO")
    add_arg("--debug", help='in debug mode', action='store_true')
    add_arg("--disable-tqdm", help='do not show progress bar', action='store_true')
    add_arg("--hadronic", help='the inputs are hadronic interactions', action='store_true')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train The GAN')
    add_arg = parser.add_argument
    add_arg("--config-file", help='use configuration file', default=None)
    add_training_args(parser)

    args = parser.parse_args()

    config = vars(args)
    if args.config_file:
        add_config = GanUtils.load_yaml(args.config_file)
        config.update(**add_config)

    config['num_epochs'] = config['max_epochs']
    GanUtils.save_configurations(config)
    # Set python level verbosity
    logging.set_verbosity(args.verbose)
    # Suppress C++ level warnings.
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    train_and_evaluate(**config)