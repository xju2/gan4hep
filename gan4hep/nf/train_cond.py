"""
Trainer for Conditional Normalizing Flow
"""
import os
import time
import re

import tensorflow as tf

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from utils import train_density_estimation_cond
from gan4hep.utils_plot import compare


def evaluate(flow_model, testing_data, cond_kwargs):
    num_samples, num_dims = testing_data.shape
    samples = flow_model.sample(num_samples, bijector_kwargs=cond_kwargs).numpy()

    distances = [
        stats.wasserstein_distance(samples[:, idx], testing_data[:, idx]) \
            for idx in range(num_dims)
    ]

    return np.average(distances), samples

def train(train_in, train_truth, test_in, testing_truth,
          flow_model, layers, lr, batch_size, max_epochs, outdir):
    base_lr = lr
    end_lr = 1e-5
    max_steps = max_epochs * train_in.shape[0] // batch_size
    learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(
        base_lr, max_steps, end_lr, power=0.5)


    # initialize checkpoints
    checkpoint_directory = "{}/checkpoints".format(outdir)
    os.makedirs(checkpoint_directory, exist_ok=True)

    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate_fn)  # optimizer
    checkpoint = tf.train.Checkpoint(optimizer=opt, model=flow_model)
    ckpt_manager = tf.train.CheckpointManager(checkpoint, checkpoint_directory, max_to_keep=None)
    latest_ckpt = ckpt_manager.latest_checkpoint
    _ = checkpoint.restore(latest_ckpt).expect_partial()
    print("Loading latest checkpoint from: {}".format(checkpoint_directory))
    if latest_ckpt:
        start_epoch = int(re.findall(r'\/ckpt-(.*)', latest_ckpt)[0]) + 1
        print("Restored from {}".format(latest_ckpt))
    else:
        start_epoch = 0
        print("Initializing from scratch.")


    AUTO = tf.data.experimental.AUTOTUNE
    training_data = tf.data.Dataset.from_tensor_slices(
        (train_in, train_truth)).batch(batch_size).prefetch(AUTO)

    img_dir = os.path.join(outdir, "imgs")
    os.makedirs(img_dir, exist_ok=True)
    log_dir = os.path.join(outdir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    csv_dir = os.path.join(outdir, "csv")
    os.makedirs(csv_dir, exist_ok=True)

    # start training
    summary_logfile = os.path.join(log_dir, 'results.txt')
    tmp_res = "# Epoch, Time, WD (Wasserstein distance), Ltr (training loss), Lte (testing loss)" 
    with open(summary_logfile, 'a') as f:
        f.write(tmp_res + "\n")

    print("idx, train loss, distance, minimum distance, minimum epoch")
    min_wdis, min_iepoch = 9999, -1
    delta_stop = 1000
    start_time = time.time()
    cond_kwargs = dict([(f"b{idx}", {"conditional_input": test_in}) for idx in range(layers)])

    for i in range(max_epochs):
        for condition,batch in training_data:
            train_loss = train_density_estimation_cond(
                flow_model, opt, batch, condition, layers)

        wdis, predictions = evaluate(flow_model, test_in, testing_truth)
        if wdis < min_wdis:
            min_wdis = wdis
            min_iepoch = i
            compare(predictions, testing_truth, img_dir, i)
            ckpt_manager.save()
        elif i - min_iepoch > delta_stop:
            break
        ckpt_manager.save(checkpoint_number = i)


        tmp_res = "* {:05d}, {:.1f}, {:.4f}, {:.4f}, {:.4f}".format(i, elapsed, wdis, avg_loss, test_loss)
        with open(summary_logfile, 'a') as f:
            f.write(tmp_res + "\n")
        print(f"{i}, {train_loss}, {wdis}, {min_wdis}, {min_iepoch}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Normalizing Flow')
    add_arg = parser.add_argument
    add_arg('filename', help='Herwig input filename')
    add_arg("outdir", help='output directory')
    add_arg("--max-evts", default=-1, type=int, help="maximum number of events")
    add_arg("--batch-size", type=int, default=512, help="batch size")
    
    args = parser.parse_args()

    from gan4hep.preprocess import herwig_angles
    from made import create_conditional_flow
    train_in, train_truth, test_in, test_truth = herwig_angles(
        args.filename, max_evts=args.max_evts)

    outdir = args.outdir
    hidden_shape = [128]*2
    layers = 10
    lr = 1e-3
    batch_size = args.batch_size
    max_epochs = 1000
    steps_per_epoch = train_in.shape[0]/batch_size
    conditional_event_shape=(4,)
    input_dim = 6 # number of parameters to be generated

    maf = create_conditional_flow(hidden_shape, layers, input_dim, conditional_event_shape)

    train(train_in, train_truth, test_in, test_truth, maf, layers, lr, batch_size, max_epochs, outdir)
