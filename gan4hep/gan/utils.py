
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
