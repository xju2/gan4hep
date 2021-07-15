
import importlib
import os
import time
import yaml

import tensorflow as tf

from gan4hep import gnn_gnn_gan as toGan
from gan4hep.gan_base import GANOptimizer
from gan4hep.graph import read_dataset, loop_dataset
from gan4hep import data_handler as DataHandler

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
    return GANOptimizer(
        gan,
        num_epcohs=num_epochs,
        disc_lr=disc_lr,
        gen_lr=gen_lr,
        with_disc_reg=with_disc_reg,
        gamma_reg=gamma_reg,
        decay_epochs=decay_epochs,
        decay_base=decay_base,
        debug=debug
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


def run_generator(gan, batch_size, filename, ngen=1000):
    dataset, n_graphs = read_dataset(filename)
    print("total {} graphs iterated with batch size of {}".format(n_graphs, batch_size))
    print('averaging {} geneveted events for each input'.format(ngen))
    test_data = loop_dataset(dataset, batch_size)

    predict_4vec = []
    truth_4vec = []
    for inputs,targets in test_data:
        input_nodes, target_nodes = DataHandler.normalize(inputs, targets, batch_size)
        
        gen_evts = []
        for igen in range(ngen):
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