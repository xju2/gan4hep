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

from utils import train_density_estimation
from utils import nll
from gan4hep.utils_plot import compare
from gan4hep.Hmumu_plots import hmumu_plot


def evaluate(flow_model, testing_data,multiplier):
    num_samples, num_dims = testing_data.shape
    
    num_samples=multiplier*num_samples

    samples = flow_model.sample(num_samples).numpy()
    distances = [
        stats.wasserstein_distance(samples[:, idx], testing_data[:, idx]) \
            for idx in range(num_dims)
    ]

    return sum(distances), samples


def train(
    train_truth, testing_truth, flow_model,
    lr, batch_size, max_epochs, outdir, xlabels,test_truth_1,gen_evts):
    """
    The primary training loop
    """
    base_lr = lr
    end_lr = 1e-5
    learning_rate_fn = tfk.optimizers.schedules.PolynomialDecay(
        base_lr, max_epochs, end_lr, power=0.5)

    # initialize checkpoints
    checkpoint_directory = "{}/checkpoints".format(outdir)
    os.makedirs(checkpoint_directory, exist_ok=True)

    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate_fn)  # optimizer
    checkpoint = tf.train.Checkpoint(optimizer=opt, model=flow_model)
    ckpt_manager = tf.train.CheckpointManager(checkpoint, checkpoint_directory, max_to_keep=None)
    _ = checkpoint.restore(ckpt_manager.latest_checkpoint).expect_partial()


    AUTO = tf.data.experimental.AUTOTUNE
    training_data = tf.data.Dataset.from_tensor_slices(
        train_truth).batch(batch_size).prefetch(AUTO)

    img_dir = os.path.join(outdir, "imgs")
    
    #summary_dir = os.path.join(log_dir, "logs")
    #summary_writer = tf.summary.create_file_writer(summary_dir) #Creates a summary file writer for the given log directory.

    #img_dir = os.path.join(log_dir, 'img')
    #os.makedirs(img_dir, exist_ok=True)

    import time
    import pathlib
    import datetime

    #Making seperate folders for each run to store plots
    #Get time and date of current run
    current_date_and_time = datetime.datetime.now()
    current_date_and_time_string = str(current_date_and_time)

    os.makedirs(img_dir, exist_ok=True)
    #Create directory path
    run_dir = os.path.join(img_dir, 'img')
    
    #Add GAN paramaters to time and date to create file name for current run
    new_run_folder=run_dir+current_date_and_time_string+' Epoch_num: '+ str(max_epochs)
    print('new_run_folder',new_run_folder)
    os.makedirs(new_run_folder, exist_ok=True)
           
    
   
    # start training
    print("idx, train loss, distance, minimum distance, minimum epoch")
    min_wdis, min_iepoch = 9999, -1
    delta_stop = 1000

    loss_list=[]
    w_list=[]
    for i in range(max_epochs):
         
        from datetime import datetime
        start_time = datetime.now()
            
        for batch in training_data:
            train_loss = train_density_estimation(flow_model, opt, batch)
            #print('train_loss',train_loss)
        wdis, predictions = evaluate(flow_model, testing_truth,gen_evts)
        end_time = datetime.now()
        print('Generation Duration: {}'.format(end_time - start_time))
        
        
        if wdis < min_wdis:
            min_wdis = wdis
            min_iepoch = i
            outname = os.path.join(img_dir, str(i))
            hmumu_plot(predictions, testing_truth, outname, xlabels,test_truth_1,new_run_folder,i)
            ckpt_manager.save()
            #save_NF(flow_model,new_run_folder)
        elif i - min_iepoch > delta_stop:
            break

        print(f"{i}, {train_loss}, {wdis}, {min_wdis}, {min_iepoch}")
        loss_list.append(train_loss)
        w_list.append(wdis)
    
    
    #Plot Log loss
    #fig, axs = plt.subplots(1, 1, figsize=(10,7), constrained_layout=True)
    #axs = axs.flatten()
    #config = dict(histtype='step', lw=2)
    
    #ax=axs
    #ax.plt(loss_list,   label='Truth',density=True,**config)
    
    
    
    #ax.set_xlabel(r"DiMuon Invarient Mass")
    #plt.yscale('log')
    #ax.legend(['Truth', 'Generator','Truth Mean','Generated SD','Truth SD','Generated Mean'])
    #plt.savefig(os.path.join(new_run_folder, 'logloss.png'.format(county)))
    #plt.close('all')
#def save_NF(flow_model,new_run_folder):
 #   print('flow_model',flow_model)
  #  print('!')
   # print('!')
    #print('!')
    #print('!')
    #flow_model_saved=flow_model
    #print(type(flow_model))
    #tmp_res = "Best Model in {} Epoch with a Wasserstein distance {:.4f}".format(best_epoch, best_wdis)
    #logging.info(tmp_res)
    
    #summary_logfile = os.path.join(new_run_folder, 'results.txt')

    #with open(summary_logfile, 'a') as f:
     #   f.write(flow_model_saved)



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
    
    args = parser.parse_args()

    from gan4hep.preprocess import herwig_angles
    from gan4hep.preprocess import dimuon_inclusive
    from made import create_flow

    #
    train_in, train_truth, test_in, test_truth,  xlabels,test_truth_1,train_truth_1= eval(args.data)(
        args.filename, max_evts=args.max_evts)
    outdir = args.outdir
    hidden_shape = [128]*2
    layers = 10
    lr = 1e-3
    batch_size = args.batch_size
    max_epochs = 10
    print('max_epochs',max_epochs)
    out_dim = train_truth.shape[1]
    gen_evts=args.multi
    maf =  create_flow(hidden_shape, layers, input_dim=out_dim, out_dim=2)
    print(maf)
    print('test_truth size',test_truth.shape)
    train(train_truth, test_truth, maf, lr, batch_size, max_epochs, outdir, xlabels,test_truth_1,gen_evts)
