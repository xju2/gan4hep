"""Trainer for Normalizing Flow
"""
import os
import tqdm
import time

import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors
tfk = tf.keras


from scipy import stats

from utils import train_density_estimation
from utils import train_density_estimation_cond
from gan4hep.utils_plot import compare


def evaluate(flow_model, testing_data):
    num_samples, num_dims = testing_data.shape
    samples = flow_model.sample(num_samples).numpy()
    distances = [
        stats.wasserstein_distance(samples[:, idx], testing_data[:, idx]) \
            for idx in range(num_dims)
    ]

    return sum(distances)/num_dims, samples


def train(train_truth, testing_truth, flow_model, layers,
    lr, batch_size, max_epochs, outdir, xlabels,
    end_lr=1e-4, power=0.5,
    disable_tqdm=False, train_in=None, test_in=None):
    """
    The primary training loop
    """
    num_steps = train_truth.shape[0] // batch_size
    base_lr = lr
    learning_rate_fn = tfk.optimizers.schedules.PolynomialDecay(
        base_lr, max_epochs*num_steps, end_lr, power=power)


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
    with_condition = False
    if train_in is not None:
        with_condition = True
        training_data = tf.data.Dataset.from_tensor_slices(
            [train_in, train_truth]).batch(batch_size).prefetch(AUTO)
    else:
        training_data = tf.data.Dataset.from_tensor_slices(
            train_truth).batch(batch_size).prefetch(AUTO)

    time_stamp = time.strftime('%Y%m%d-%H%M%S', time.localtime())
    img_dir = os.path.join(outdir, "imgs", f"{time_stamp}")
    os.makedirs(img_dir, exist_ok=True)

    # start training
    min_wdis, min_iepoch = 9999, -1
    delta_stop = 1000

    
    summary_dir = os.path.join(outdir, "logs", f"{time_stamp}")
    summary_writer = tf.summary.create_file_writer(summary_dir)
    summary_logfile = os.path.join(summary_dir, f'results_{time_stamp}.txt')

    cond_kwargs = dict([(f"b{idx}", {"conditional_input": test_in}) for idx in range(layers)])
    with tqdm.trange(max_epochs, disable=disable_tqdm) as t0:
        for epoch in t0:
            
            tot_loss = []
            ## different training loop without or without conditional variables
            if with_condition:
                for condition,batch in training_data:
                    train_loss = train_density_estimation_cond(
                        flow_model, opt, batch, condition, cond_kwargs)
                    tot_loss.append(train_loss)
            else:
                for batch in training_data:
                    train_loss = train_density_estimation(flow_model, opt, batch)
                    tot_loss.append(train_loss)

            tot_loss = np.array(tot_loss)
            avg_loss = np.sum(tot_loss, axis=0) / tot_loss.shape[0]

            log_dict = dict(t_loss=avg_loss)
            wdis, predictions = evaluate(flow_model, testing_truth)


            with summary_writer.as_default():
                tf.summary.experimental.set_step(epoch)
                tf.summary.scalar("avg_wasserstein_dis",
                    wdis, description="average wasserstein distance")

                for key,val in log_dict.items():
                    tf.summary.scalar(key, val)

            if wdis < min_wdis:
                ckpt_manager.save()
                min_wdis = wdis
                min_iepoch = epoch
                outname = os.path.join(img_dir, f"{epoch}.png")
                compare(predictions, testing_truth, outname, xlabels)
                with open(summary_logfile, 'a') as f:
                    f.write(", ".join(["{:.4f}".format(x) 
                        for x in [min_wdis, min_iepoch]]) + '\n')
            elif epoch - min_iepoch > delta_stop:
                print(f"Seen no improvement after training for more than {delta_stop} epochs")
                print("stop the training")
                break

            t0.set_postfix(**log_dict, BestD=min_wdis, BestE=min_iepoch)

if __name__ == '__main__':
    import argparse
    from gan4hep import io
    from made import create_flow


    parser = argparse.ArgumentParser(description='Normalizing Flow')
    add_arg = parser.add_argument
    add_arg('filename', help='Herwig input filename')
    add_arg("outdir", help='output directory')
    add_arg("--max-evts", default=-1, type=int, help="maximum number of events")
    add_arg("--batch-size", type=int, default=512, help="batch size")
    add_arg("--data", default='herwig_angles', choices=io.__all__)

    ## hyperpaprameters
    add_arg("--lr", type=float, default=0.001, help="learning rate")
    add_arg("--end-lr", type=float, default=1e-4, help="end learning rate")
    add_arg("--power", type=float, default=0.5, help="learning rate decay power")

    add_arg("--max-epochs", type=int, default=2000, help="maximum number of epochs")
    add_arg("--hidden-shape", type=int, nargs='+', default=[128, 128], help="hidden shape")
    add_arg("--num-layers", type=int, default=10, help="number of layers")
    
    args = parser.parse_args()

    train_in, train_truth, test_in, test_truth, xlabels = getattr(io, args.data)(
        args.filename, max_evts=args.max_evts)

    outdir = args.outdir
    hidden_shape = args.hidden_shape
    layers = args.num_layers
    lr = args.lr
    batch_size = args.batch_size
    max_epochs = args.max_epochs
    out_dim = train_truth.shape[1]

    maf =  create_flow(hidden_shape, layers, input_dim=out_dim)
    print(maf)
    train(train_truth, test_truth, maf, layers, lr,
        batch_size, max_epochs, outdir, xlabels,
        end_lr=args.end_lr, power=args.power)
