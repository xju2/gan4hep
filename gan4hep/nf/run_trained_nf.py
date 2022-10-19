"""Trainer for Normalizing Flow
"""
import os


import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors
tfk = tf.keras

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import pandas as pd
from utils import train_density_estimation
from utils import nll
from gan4hep.utils_plot import compare
import glob
import os



def train(
    train_truth, testing_truth, flow_model,
    lr, batch_size, max_epochs, outdir, xlabels,test_truth_1,gen_evts,num_gen_evts,full_truth,truth_data_1):



    #Create timestamp for generated data
    import time
    import pathlib
    import datetime


    # Making seperate folders for each run to store plots
    # Get time and date of current run
    current_date_and_time = datetime.datetime.now()
    current_date_and_time_string = str(current_date_and_time)

    #Recording how long it takes to generate data
    from datetime import datetime
    start_time = datetime.now()

    #Define learning Rate
    base_lr = lr
    end_lr = 1e-5
    learning_rate_fn = tfk.optimizers.schedules.PolynomialDecay(
        base_lr, max_epochs, end_lr, power=0.5)

    # initialize checkpoints
    checkpoint_directory = "{}/checkpoints2".format(outdir)
    os.makedirs(checkpoint_directory, exist_ok=True)

    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate_fn)  # optimizer
    checkpoint = tf.train.Checkpoint(optimizer=opt, model=flow_model)
    ckpt_manager = tf.train.CheckpointManager(checkpoint, checkpoint_directory, max_to_keep=None)
    _ = checkpoint.restore(ckpt_manager.latest_checkpoint).expect_partial()



    #Generate new data
    print(flow_model)
    num_samples, num_dims = testing_truth.shape
    num_samples=num_gen_evts
    print('Number of Generated Events: ', num_samples)
    samples = flow_model.sample(num_samples).numpy()
    predictions=samples

    end_time = datetime.now()
    print('Generation Duration for new events : {}'.format(end_time - start_time))

    # Apply Inverse Scaler to get original value ranges back
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(-1, 1))
    truth_data_1 = scaler.fit_transform(truth_data_1)
    truths = scaler.inverse_transform(testing_truth)
    predictions = scaler.inverse_transform(predictions)

    # Create folder to store generated data if it doesn't already exist
    if os.path.exists('Generated_Data') == False:
        os.mkdir('Generated_Data')
    filename=str('predictions_'+ current_date_and_time_string+'_num_of_events_'+str(num_gen_evts)+'.npy')
    filename2='truths.npy'

    # Save to folder
    np.save(os.path.join('Generated_Data', filename ), predictions)
    np.save(os.path.join('Generated_Data', filename2), full_truth)

    # Plot of generated Variables
    num_of_variables = 9
    fig, axs = plt.subplots(1, 9, figsize=(50, 10), constrained_layout=True)
    axs = axs.flatten()
    # config = dict(histtype='step', lw=2)
    config = dict(histtype='step', lw=2)
    i = 0
    for i in range(9):
        idx = i
        ax = axs[idx]
        ax.hist(full_truth[:, int(idx)], bins=20, range=[min(predictions[:, idx]), max(predictions[:, idx])],
                label='Truth', density=True, **config)
        ax.hist(predictions[:, idx], bins=20, range=[min(predictions[:, idx]), max(predictions[:, idx])],
                label='Generator', density=True, **config)
        #ax.set_xlabel(xlabels_extra[i], fontsize=16)
        ax.legend(['Truth', 'Generator'], loc=3)
        # ax.set_yscale('log')

        # Save Figures
    plt.savefig(os.path.join('Generated_Data', 'image{:04d}.png'))
    plt.close('all')
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Normalizing Flow')
    add_arg = parser.add_argument
    add_arg('filename', help='Herwig input filename')
    add_arg("outdir", help='output directory')
    add_arg("--max-evts", default=-1, type=int, help="maximum number of events")
    add_arg("--batch-size", type=int, default=512, help="batch size")
    add_arg("--multi", type=int, default=1, help="Number times more of generated events")
    add_arg("--num-gen-evts", type=int, default=10000, help="Number of events generated")
    add_arg("--data", default='dimuon_inclusive',
            choices=['herwig_angles', 'dimuon_inclusive', 'herwig_angles2'])

    args = parser.parse_args()

    from gan4hep.preprocess import herwig_angles
    from gan4hep.preprocess import dimuon_inclusive
    from made import create_flow

    train_in, train_truth, test_in, test_truth, xlabels, test_truth_1, train_truth_1,full_data,truth_data_1 = eval(args.data)(
        args.filename, max_evts=args.max_evts)


    num_gen_evts=args.num_gen_evts


    #If number of generated events is less than the length of the true dataset
    #if num_gen_evts<=2100000:
     #   full_data = full_data[:num_gen_evts]

    outdir = args.outdir
    hidden_shape = [128] * 2
    layers = 10
    lr = 1e-3
    batch_size = args.batch_size
    max_epochs = 90
    out_dim = train_truth.shape[1]
    gen_evts = args.multi
    max_evts=args.max_evts

    maf,sample = create_flow(max_evts,hidden_shape, layers=10, input_dim=out_dim)
    train(train_truth, test_truth, maf, lr, batch_size, max_epochs, outdir, xlabels, test_truth_1, gen_evts,num_gen_evts,full_data,truth_data_1)
