
import importlib
from operator import truth
import os
import time
import yaml

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import tensorflow as tf

from gan4hep.utils_plot import compare


def import_model(gan_name):
    gan_module = importlib.import_module("gan4hep."+gan_name)
    return gan_module

def create_gan(gan_type, noise_dim, batch_size, layer_size, num_layers,
    num_processing_steps=None, **kwargs):
    gan_model = import_model(gan_type)
    
    if num_processing_steps and gan_type == "gnn_gnn_gan":
        gan = gan_model.GAN(
            noise_dim, batch_size, layer_size, num_layers,
            num_processing_steps, name=gan_type)
    else:
        gan = gan_model.GAN(
            noise_dim, batch_size, latent_size=layer_size,
            num_layers=num_layers, name=gan_type)

    return gan

def create_optimizer(gan, num_epochs, disc_lr, gen_lr, with_disc_reg, gamma_reg,
        decay_epochs, decay_base, debug, **kwargs):

    from gan4hep.gan_base import GANOptimizer
    return GANOptimizer(
        gan,
        num_epcohs=num_epochs,
        disc_lr=disc_lr,
        gen_lr=gen_lr,
        with_disc_reg=with_disc_reg,
        gamma_reg=gamma_reg,
        decay_epochs=decay_epochs,
        decay_base=decay_base,
        debug=debug,
    )

def get_ckptdir(output_dir, **kwargs):
    return os.path.join(output_dir, "checkpoints")

def load_gan(config_name: str):
    if not os.path.exists(config_name):
        raise RuntimeError(f"{config_name} does not exist")

    config = load_yaml(config_name)
    gan = create_gan(**config)
    optimizer = create_optimizer(gan, **config)

    ckpt_dir = get_ckptdir(**config)

    checkpoint = tf.train.Checkpoint(optimizer=optimizer, gan=gan)
    ckpt_manager = tf.train.CheckpointManager(
        checkpoint, directory=ckpt_dir, max_to_keep=10, keep_checkpoint_every_n_hours=1)
    _ = checkpoint.restore(ckpt_manager.latest_checkpoint)
    return gan


def run_generator(gan, batch_size, filename, hadronic, ngen=1):
    from gan4hep.graph import read_dataset, loop_dataset
    from gan4hep import data_handler as DataHandler

    dataset, n_graphs = read_dataset(filename)
    print("total {} graphs iterated with batch size of {}".format(n_graphs, batch_size))
    print('averaging {} geneveted events for each input'.format(ngen))
    test_data = loop_dataset(dataset, batch_size)

    predict_4vec = []
    truth_4vec = []
    for inputs,targets in test_data:
        input_nodes, target_nodes = DataHandler.normalize(inputs, targets, batch_size, hadronic=hadronic)
        
        gen_evts = []
        for _ in range(ngen):
            gen_graph = gan.generate(input_nodes, is_training=False)
            gen_evts.append(gen_graph)
        
        gen_evts = tf.reduce_mean(tf.stack(gen_evts), axis=0)
        
        predict_4vec.append(tf.reshape(gen_evts, [batch_size, -1, 4]))
        truth_4vec.append(tf.reshape(target_nodes, [batch_size, -1, 4]))
        
    predict_4vec = tf.concat(predict_4vec, axis=0)
    truth_4vec = tf.concat(truth_4vec, axis=0)
    return predict_4vec, truth_4vec

def load_yaml(filename):
    with open(filename) as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def compare_two_dicts(dict1, dict2):
    """
    Return True if they are identical, False otherwise
    """
    res = True
    for key,value in dict1.items():
        if key not in dict2 or value != dict2[key]:
            res = False
            break
    return res

def get_file_ctime(path):
    """
    Return the creation time of the path in string format
    """
    ct = os.path.getctime(path)
    time_stamp = time.strftime('%Y%m%d-%H%M%S', time.gmtime(ct))
    return time_stamp

def save_configurations(config: dict):
    outname = "config_"+config['output_dir'].replace("/", '_')+'.yml'
    if os.path.exists(outname):
        existing_config = load_yaml(outname)
        if compare_two_dicts(existing_config, config):
            return
        back_name = outname.replace(".yml", "")
        back_name += "_"+get_file_ctime(outname) + '.yml'
        os.rename(outname, back_name)

    with open(outname, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)



def log_metrics(
        summary_writer,
        predict_4vec: np.ndarray,
        truth_4vec: np.ndarray,
        step: int,
        **kwargs):

    with summary_writer.as_default():
        tf.summary.experimental.set_step(step)
        # plot the eight variables and resepctive Wasserstein distance (i.e. Earch Mover Distance)
        # Use the Kolmogorov-Smirnov test, 
        # it turns a two-sided test for the null hypothesis that the two distributions
        # are drawn from the same continuous distribution.
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.combine_pvalues.html#scipy.stats.combine_pvalues
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ks_2samp.html#scipy.stats.ks_2samp
        # https://docs.scipy.org/doc/scipy/reference/stats.html#statistical-distances
        distances = []

        for icol in range(predict_4vec.shape[1]):
            dis = stats.wasserstein_distance(predict_4vec[:, icol], truth_4vec[:, icol])
            _, pvalue = stats.ks_2samp(predict_4vec[:, icol], truth_4vec[:, icol])
            if pvalue < 1e-7: pvalue = 1e-7
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

        if kwargs is not None:
            # other metrics to be recorded.
            for key,val in kwargs.items():
                tf.summary.scalar(key, val)

    return tot_wdis, comb_pvals, tot_edis, tot_mse


def evaluate(model, datasets):
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
    predictions = []
    truths = []
    for data in datasets:
        test_input, test_truth = data
        predictions.append(model(test_input, training=False))
        truths.append(test_truth)

    predictions = tf.concat(predictions, axis=0).numpy()
    truths = tf.concat(truths, axis=0).numpy()
    return predictions, truths


def generate_and_save_images(model, epoch, datasets, summary_writer, img_dir, xlabels, **kwargs) -> float:
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
    predictions = []
    truths = []
    for data in datasets:
        test_input, test_truth = data
        predictions.append(model(test_input, training=False))
        truths.append(test_truth)

    predictions = tf.concat(predictions, axis=0).numpy()
    truths = tf.concat(truths, axis=0).numpy()

    outname = os.path.join(img_dir, str(epoch))
    compare(predictions, truths, outname, xlabels)

    if summary_writer:
        return log_metrics(summary_writer, predictions, truths, epoch, **kwargs)[0]
    else:
        return -9999.