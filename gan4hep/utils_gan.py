


import importlib
import os
import time
from tensorboard.summary._tf import summary
import yaml

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

import tensorflow as tf

from gan4hep import gnn_gnn_gan as toGan
from gan4hep.gan_base import GANOptimizer
from gan4hep.graph import read_dataset, loop_dataset
from gan4hep import data_handler as DataHandler
from gan4hep.gan.gan import GAN

from pylorentz import Momentum4


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


def run_generator(gan, batch_size, filename, hadronic, ngen=1):
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



def generate_and_save_images(model,epoch,datasets,summary_writer,img_dir,new_run_folder,loss_all_epochs_0,loss_all_epochs_1,discriminator,gen_accuracy,accuracy_list, **kwargs):
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
    predictions = []
    truths = []
    
    #datasets is testing_data a combination of truth and in data split into batches but not shuffled
    for data in datasets:
        test_input, test_truth = data

        predictions.append(model(test_input, training=False)) # putting the gaussian generated noise through the 
        truths.append(test_truth)

    #predictions contains generated noise from test_input and truths contain data from the original output file
    predictions = tf.concat(predictions, axis=0).numpy()
    truths = tf.concat(truths, axis=0).numpy()
    
    #Calculate invarient di-muon mass for each event
    mumu_true,mumu_pred=mumu_invariant_mass(truths,predictions)
    
    #Get probability that each event is true or generated
    predictions_d=discriminator(predictions)
    truths_d=discriminator(truths)
    
    #Convert probabilities into binaries
    predictions_binary = [0 if i <=0.5 else 1 for i in predictions_d]
    truths_binary = [0 if i <=0.5 else 1 for i in truths_d]

    #Calculate average of each array for an epoch
    predictions_avg=np.mean(predictions_binary)
    truths_avg=np.mean(truths_binary)

    print(len(predictions),'Length of Generated Dataset')
    print(len(truths),'Length of Truth Dataset')

    print(predictions_binary.count(1), 'Generated Data Identified as True Data')
    print(predictions_binary.count(0), 'Generated Data Identified as Generated Data')
    print(truths_binary.count(1), 'True Data Identified as True Data')
    print(truths_binary.count(0), 'True Data Identified as Generated Data')
    
    #Calculate accuracy for the discriminator
    acc=(truths_binary.count(1)+predictions_binary.count(0))/(truths_binary.count(1)+predictions_binary.count(0)+truths_binary.count(0)+predictions_binary.count(1))

        
    #Calculate accuracy for the generator
    gen_acc=predictions_binary.count(1)/(predictions_binary.count(1)+predictions_binary.count(0))

    #Append to relevent lists
    gen_accuracy.append(gen_acc)
    accuracy_list.append(acc)

    
    #Creating Plots
    fig, axs = plt.subplots(1, 7, figsize=(20, 6), constrained_layout=True)
    axs = axs.flatten()

    config = dict(histtype='step', lw=2)

    #Muons_PT_Lead:Muons_Eta_Lead:Muons_Phi_Lead:Muons_PT_Sub:Muons_Eta_Sub:Muons_Phi_Sub
    
    #Plot 1
    idx=0
    ax = axs[idx]
    x_range = [-1, 1]
    
    yvals, _, _ = ax.hist(truths[:, idx], bins=40,  label='Truth', **config)
    max_y = np.max(yvals) * 1.1
    ax.hist(predictions[:, idx], bins=40, range=x_range, label='Generator', **config)
    ax.set_xlabel(r"Muons_PT_Lead")
    #ax.set_ylim(0, max_y)
    ax.legend(['Truth', 'Generator'])
    #ax.set_yscale('log')

    # Plot 2
    idx=1
    ax = axs[idx]
    yvals, _, _ = ax.hist(truths[:, idx],  bins=40, label='Truth', **config)
    max_y = np.max(yvals) * 1.1
    ax.hist(predictions[:, idx], bins=40, range=x_range, label='Generator', **config)
    ax.set_xlabel(r"Muons_Eta_Lead")
    ax.legend(['Truth', 'Generator'])
    #ax.set_ylim(0, max_y)
    #ax.set_yscale('log')

    # plot 3
    idx=2
    ax = axs[idx]
    x_range = [-1, 1]
    
    yvals, _, _ = ax.hist(truths[:, idx], bins=40,  label='Truth', **config)
    max_y = np.max(yvals) * 1.1
    ax.hist(predictions[:, idx], bins=40, range=x_range, label='Generator', **config)
    ax.set_xlabel(r"Muons_Phi_Lead")
    ax.legend(['Truth', 'Generator'])

    # plot 4
    idx=3
    ax = axs[idx]
    yvals, _, _ = ax.hist(truths[:, idx],  bins=40,  label='Truth', **config)
    max_y = np.max(yvals) * 1.1
    ax.hist(predictions[:, idx], bins=40, range=x_range, label='Generator', **config)
    ax.set_xlabel(r"Muons_PT_Sub")
    ax.legend(['Truth', 'Generator'])
 
    # plot 5
    idx=4
    ax = axs[idx]
    x_range = [-1, 1]
    
    yvals, _, _ = ax.hist(truths[:, idx], bins=40,  label='Truth', **config)
    max_y = np.max(yvals) * 1.1
    ax.hist(predictions[:, idx], bins=40, range=x_range, label='Generator', **config)
    ax.set_xlabel(r"Muons_Eta_Sub")
    ax.legend(['Truth', 'Generator'])

    # plot 6
    idx=5
    ax = axs[idx]
    yvals, _, _ = ax.hist(truths[:, idx],  bins=40,  label='Truth', **config)
    max_y = np.max(yvals) * 1.1
    ax.hist(predictions[:, idx], bins=40, range=x_range, label='Generator', **config)
    ax.set_xlabel(r"Muons_Phi_Sub")
    ax.legend(['Truth', 'Generator'])
    
    # plot 7
    
    ax = axs[6]
    yvals, _, _ = ax.hist(mumu_true,  bins=80,  label='Truth', **config,range=[0, 3.0])
    #max_y = np.max(yvals) * 1.1
    ax.hist(mumu_pred, bins=80, range=[0, 5.0], label='Generator', **config)
    ax.set_xlabel(r"Mu_Mu_Invariant_Mass")
    ax.legend(['Truth', 'Generator'])
    
    # plt.legend()
    plt.savefig(os.path.join(new_run_folder, 'image_at_epoch_{:04d}.png'.format(epoch)))
    plt.close('all')
    
    

    if summary_writer:
        return log_metrics(summary_writer, predictions, truths, epoch, **kwargs)[0],accuracy_list,gen_accuracy
    else:
        return -9999.


def generate_and_save_images_end_of_run(epoch,img_dir,new_run_folder,loss_all_epochs_0,loss_all_epochs_1,accuracy_list,gen_accuracy,best_was_dist):  
    
        
    # plot loss, accuracy, and best wasserstein distance plots

    plt.plot(loss_all_epochs_0, color='blue')
    plt.xlabel('Epoch Number')
    plt.ylabel('Discriminator Loss')
    plt.savefig(os.path.join(new_run_folder, 'log_loss_curve_0.png'.format(epoch)))
    plt.close('all')

    plt.plot(loss_all_epochs_1, color='orange')
    plt.xlabel('Epoch Number')
    plt.ylabel('Generator Loss')
    plt.savefig(os.path.join(new_run_folder, 'log_loss_curve_1.png'.format(epoch)))
    plt.close('all')

    plt.plot(loss_all_epochs_0,  label='Discriminator Loss')
    plt.plot(loss_all_epochs_1, label='Generator Loss')
    plt.xlabel('Epoch Number')
    plt.ylabel('Loss')
    plt.legend(loc="best")
    plt.savefig(os.path.join(new_run_folder, 'log_loss_curve_comb.png'.format(epoch)))
    plt.close('all')

    plt.plot(accuracy_list)
    plt.xlabel('Epoch Number')
    plt.ylabel('Discriminator Accuracy')
    plt.legend(loc="best")
    plt.savefig(os.path.join(new_run_folder, 'd_accuracy.png'.format(epoch)))
    plt.close('all')

    plt.plot(gen_accuracy)
    plt.xlabel('Epoch Number')
    plt.ylabel('Generator Accuracy')
    plt.legend(loc="best")
    plt.savefig(os.path.join(new_run_folder, 'g_accuracy.png'.format(epoch)))
    plt.close('all')

    plt.plot(accuracy_list,  label='Discriminator Accuracy')
    plt.plot(gen_accuracy, label='Generator Accuracy')
    plt.xlabel('Epoch Number')
    plt.ylabel('Accuracy')
    plt.legend(loc="best")
    plt.savefig(os.path.join(new_run_folder, 'total_acc.png'.format(epoch)))
    plt.close('all')


    plt.plot(best_was_dist, color='blue')
    plt.xlabel('Epoch Number')
    plt.ylabel('Best Wasserstein Distance')
    plt.savefig(os.path.join(new_run_folder, 'wasserstein_dist.png'.format(epoch)))
    plt.close('all')
    
    
    
    
def mumu_invariant_mass(truths,predictions):
    
    #Define muon mass and create two arrays filled with this value
    m_u=1.883531627e-28
    masses_lead = np.full((len(truths[:,0]), 1), m_u)
    masses_sub = np.full((len(truths[:,0]), 1), m_u)
    
    #Take each column from the tru and generated data and rename to their parameter type
    pts_lead_true = np.array(truths[:, 0]).flatten()
    etas_lead_true = np.array(truths[:, 1]).flatten()
    phis_lead_true = np.array(truths[:, 2]).flatten()
    pts_sub_true = np.array(truths[:, 3]).flatten()
    etas_sub_true =np.array(truths[:, 4]).flatten()
    phis_sub_true = np.array(truths[:, 5]).flatten()
    
    pts_lead_gen = np.array(predictions[:, 0]).flatten()
    etas_lead_gen = np.array(predictions[:, 1]).flatten()
    phis_lead_gen = np.array(predictions[:, 2]).flatten()
    pts_sub_gen = np.array(predictions[:, 3]).flatten()
    etas_sub_gen = np.array(predictions[:, 4]).flatten()
    phis_sub_gen = np.array(predictions[:, 5]).flatten()
    
    #Create lists for 4 vector values
    muon_lead_true=[]
    muon_sub_true=[]
    muon_lead_gen=[]
    muon_sub_gen=[]
    parent_true=[]
    parent_gen=[]
    
    #Create lists for invarient mass values
    mass_true=[]
    mass_gen=[]

    for i in range(len(truths)):
        #Use pylorentz to define 4 momentum arrays for each event
        muon_lead_true.append(Momentum4.m_eta_phi_pt(masses_lead[i], etas_lead_true[i], phis_lead_true[i], pts_lead_true[i]))
        muon_sub_true.append(Momentum4.m_eta_phi_pt(masses_sub[i], etas_sub_true[i], phis_sub_true[i], pts_sub_true[i]))
        muon_lead_gen.append(Momentum4.m_eta_phi_pt(masses_lead[i], etas_lead_gen[i], phis_lead_gen[i], pts_lead_gen[i]))
        muon_sub_gen.append(Momentum4.m_eta_phi_pt(masses_sub[i], etas_sub_gen[i], phis_sub_gen[i], pts_sub_gen[i]))
        
        #Calculate the Higgs boson 4 vector
        parent_true.append(muon_lead_true[i] + muon_sub_true[i])
        parent_gen.append(muon_lead_gen[i] + muon_sub_gen[i])
        
        #Retrieve the Higgs Mass
        mass_true.append(parent_true[i].m)
        mass_gen.append(parent_gen[i].m)

    #Add mass arrays from each batch?    
    mass_true=np.concatenate( mass_true, axis=0 )
    mass_gen=np.concatenate( mass_gen, axis=0 )
  
    return mass_true,mass_gen 
    
