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

import time
import pathlib
import datetime
from datetime import datetime



def evaluate(flow_model, testing_data,gen_evts):
    num_samples, num_dims = testing_data.shape
    num_samples=num_samples*gen_evts
    print('Number of samples generated: ',num_samples)
    samples = flow_model.sample(num_samples).numpy()
    distances = [
        stats.wasserstein_distance(samples[:, idx], testing_data[:, idx]) \
        for idx in range(num_dims)
    ]
    return sum(distances), samples




def save_data(test_truth_1,w_list,loss_list,best_true_data,best_gen_data,wdis,new_run_folder):

    # Apply Inverse Scaler to get original values back
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(-1, 1))
    test_truth_1 = scaler.fit_transform(test_truth_1)
    truths = scaler.inverse_transform(best_true_data)
    predictions = scaler.inverse_transform(best_gen_data)
    print('Best Wasserstein Distance: ', wdis)

    # Create temporary folder for using the plotticng program
    if os.path.exists('Temp_Data') == False:
        os.mkdir('Temp_Data')

    # Save data to run data folder
    np.save(os.path.join(new_run_folder, 'truths.npy'), truths)
    np.save(os.path.join(new_run_folder, 'predictions.npy'), truths)
    np.save(os.path.join(new_run_folder, 'loss_list.npy'), loss_list)
    np.save(os.path.join(new_run_folder, 'w_list.npy'), w_list)

    # Save to temporary folder
    np.save(os.path.join('Temp_Data', 'truths.npy'), truths)
    np.save(os.path.join('Temp_Data', 'predictions.npy'), predictions)
    np.save(os.path.join('Temp_Data', 'loss_list.npy'), loss_list)
    np.save(os.path.join('Temp_Data', 'w_list.npy'), w_list)

    # Saving file name to run data folder
    with open('Temp_Data' + "/filename.txt", "w") as f:
        f.write(new_run_folder)

    with open(new_run_folder + "/filename.txt", "w") as f:
        f.write(new_run_folder)



def train(
        sample,train_truth, testing_truth, flow_model,
        lr, batch_size, max_epochs, outdir, xlabels, test_truth_1, gen_evts, start_time_full):

    """
    The primary training loop
    """
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

    AUTO = tf.data.experimental.AUTOTUNE
    training_data = tf.data.Dataset.from_tensor_slices(
        train_truth).batch(batch_size).prefetch(AUTO)

    img_dir = os.path.join(outdir, "imgs")

    # summary_dir = os.path.join(log_dir, "logs")
    # summary_writer = tf.summary.create_file_writer(summary_dir) #Creates a summary file writer for the given log directory.

    # img_dir = os.path.join(log_dir, 'img')
    # os.makedirs(img_dir, exist_ok=True)

    # Making seperate folders for each run to store plots
    # Get time and date of current run
    current_date_and_time = datetime.now()
    current_date_and_time_string = str(current_date_and_time)

    os.makedirs(img_dir, exist_ok=True)
    # Create directory path
    run_dir = os.path.join(img_dir, 'img')

    # Add GAN paramaters to time and date to create file name for current run
    new_run_folder = run_dir + current_date_and_time_string + ' Epoch_num: ' + str(max_epochs)
    print('New Run Folder Created: ', new_run_folder)
    os.makedirs(new_run_folder, exist_ok=True)

    # start training
    print("idx, train loss, distance, minimum distance, minimum epoch")
    min_wdis, min_iepoch = 9999, -1
    delta_stop = 1000

    time_list = []
    loss_list = []
    w_list = []

    for i in range(max_epochs):

        start_time = datetime.now()

        for batch in training_data:
            train_loss = train_density_estimation(flow_model, opt, batch)
        wdis, predictions = evaluate(flow_model, testing_truth,gen_evts)

        end_time = datetime.now()
        print('Generation Duration for epoch ' + str(i) + ': {}'.format(end_time - start_time))
        time_state = 'Generation Duration for epoch ' + str(i) + ': {}'.format(end_time - start_time)
        time_list.append(time_state)


        # Original Variables plot for each epoch

        num_var=len(testing_truth[1,:])
        fig, axs = plt.subplots(1, num_var, figsize=(50, 10), constrained_layout=True)
        axs = axs.flatten()
        # config = dict(histtype='step', lw=2)
        config = dict(histtype='step', lw=2)
        j = 0
        county=i

        for j in range(num_var):
            idx = j
            ax = axs[idx]
            yvals, _, _ = ax.hist(testing_truth[:, idx], bins=40, range=[min(testing_truth[:, idx]), max(testing_truth[:, idx])], label='Truth',density=True, **config)
            max_y = np.max(yvals) * 1.1
            ax.hist(predictions[:, idx], bins=40,range=[min(testing_truth[:, idx]), max(testing_truth[:, idx])], label='Generator', density=True, **config)
            #ax.set_xlabel(xlabels_extra[i], fontsize=16)
            ax.legend(['Truth', 'Generator'], loc=3)
            # ax.set_yscale('log')

            # Save Figures
        plt.savefig(os.path.join(new_run_folder, 'image_at_epoch_{:04d}.png'.format(county)))
        plt.close('all')



        if wdis < min_wdis:
            min_wdis = wdis
            min_iepoch = i
            outname = os.path.join(img_dir, str(i))
            ckpt_manager.save()
            w_list.append(float(f"{wdis}"))
            best_gen_data=predictions
            best_true_data=testing_truth

            #Save Data for an improved model
            save_data(test_truth_1,w_list,loss_list,best_true_data,best_gen_data,wdis,new_run_folder)

        elif i - min_iepoch > delta_stop:
            break
        else:
            w_list.append(float(f"{wdis}"))
        print(f"{i}, {train_loss}, {wdis}, {min_wdis}, {min_iepoch}")

        loss_list.append(float(f"{train_loss}"))


    end_time_full = datetime.now()
    full_time_state = 'Total Generation Duration: {}'.format(end_time_full - start_time_full)
    time_list.append(full_time_state)
    with open(new_run_folder + "/time.txt", "w") as f:
        f.write(str(time_list))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Normalizing Flow')
    add_arg = parser.add_argument
    add_arg('filename', help='Herwig input filename')
    add_arg("outdir", help='output directory')
    add_arg("--max-evts", default=-1, type=int, help="maximum number of events")
    add_arg("--batch-size", type=int, default=512, help="batch size")
    add_arg("--multi", type=int, default=1, help="Number times more of generated events")
    add_arg("--data", default='dimuon_inclusive',
            choices=['herwig_angles', 'dimuon_inclusive', 'herwig_angles2'])


    start_time_full = datetime.now()

    args = parser.parse_args()

    from gan4hep.preprocess import herwig_angles
    from gan4hep.preprocess import dimuon_inclusive
    from made import create_flow

    train_in, train_truth, test_in, test_truth, xlabels, test_truth_1, train_truth_1,full_data,truth_data_1 = eval(args.data)(
        args.filename, max_evts=args.max_evts)

    max_evts = args.max_evts
    outdir = args.outdir
    hidden_shape = [128] * 2
    layers = 10
    lr = 1e-3
    batch_size = args.batch_size
    max_epochs = 3000
    print('Number of Epochs: ', max_epochs)

    out_dim = train_truth.shape[1]
    gen_evts = args.multi

    maf,sample = create_flow(max_evts,hidden_shape, layers, input_dim=out_dim)

    train(sample,train_truth, test_truth, maf, lr, batch_size, max_epochs, outdir, xlabels, test_truth_1, gen_evts,
          start_time_full)
