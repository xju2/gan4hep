
import importlib
import os

import tensorflow as tf

from gan4hep import gnn_gnn_gan as toGan
from gan4hep.gan_base import GANOptimizer
from gan4hep.graph import read_dataset, loop_dataset
from gan4hep import reader

gan_types = ['mlp_gan', 'rnn_mlp_gan', 'rnn_rnn_gan', 'gnn_gnn_gan']
def import_model(gan_name):
  gan_module = importlib.import_module("gan4hep."+gan_name)
  return gan_module


def load_model_from_ckpt(gan_type, noise_dim, batch_size, ckpt_dir):
  toGan = import_model(gan_type)
  gan = toGan.GAN(noise_dim, batch_size)
  optimizer = GANOptimizer(gan)

  ckpt_dir = os.path.join(ckpt_dir, "checkpoints")
  checkpoint = tf.train.Checkpoint(
      optimizer=optimizer,
      gan=gan)

  ckpt_manager = tf.train.CheckpointManager(checkpoint, directory=ckpt_dir,
                                            max_to_keep=5, keep_checkpoint_every_n_hours=8)
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
      input_nodes, target_nodes = reader.normalize(inputs, targets, batch_size)
      
      gen_evts = []
      for igen in range(ngen):
          gen_graph = gan.generate(input_nodes)
          gen_evts.append(gen_graph)
      
      gen_evts = tf.reduce_mean(tf.stack(gen_evts), axis=0)
      
      predict_4vec.append(tf.reshape(gen_evts, [batch_size, -1, 4]))
      truth_4vec.append(tf.reshape(target_nodes, [batch_size, -1, 4]))
      
  predict_4vec = tf.concat(predict_4vec, axis=0)
  truth_4vec = tf.concat(truth_4vec, axis=0)
  return predict_4vec, truth_4vec