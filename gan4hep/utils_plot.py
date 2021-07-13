import matplotlib.pyplot as plt
import numpy as np
import yaml

def add_mean_std(array, x, y, ax, color='k', dy=None, digits=2, fontsize=12, with_std=True):
    this_mean, this_std = np.mean(array), np.std(array)
    if dy is None:
        dy = y * 0.1
    ax.text(x, y, "mean: {0:.{1}f}".format(this_mean, digits), color=color, fontsize=12)
    if with_std:
        ax.text(x, y-dy, "std: {0:.{1}f}".format(this_std, digits), color=color, fontsize=12)
    return ax
        
        
def array2hist(array, ax=None, with_mean_std=True, bins=100, **kwargs):
    if ax is None:
        _, ax = plt.subplots(figsize=(6,6))
    entries, _, _ = ax.hist(array, bins=bins, **kwargs)
    if with_mean_std:
        min_x, max_x = np.min(array), np.max(array)
        x = 0.5*(max_x + min_x)
        y = np.max(entries)*0.8
        add_mean_std(array, x, y, ax)
    return ax


def view_particle_4vec(particles, axs=None, labels=None, outname=None, **kwargs):
    """
    make histograms of the 4 vectors of the particles
    Inputs:
        particles: 2D arrays (num_particles x 4vectors)
        labels: naming of the 4vectors, default ['$P_x$ [GeV]', '$P_y$ [GeV]', '$P_z$ [GeV]', "$E$ [GeV]"]
    Return:
        ax
    """
    if labels is None:
        labels = ['$P_x$ [GeV]', '$P_y$ [GeV]', '$P_z$ [GeV]', "$E$ [GeV]"]

    if axs is None or len(axs) != 4:
        fig, axs = plt.subplots(2,2, figsize=(10,10))
        axs = axs.flatten()
    
    for idx in range(4):
        array2hist(particles[:, idx], axs[idx], **kwargs)
        axs[idx].set_xlabel(labels[idx])

    if outname is not None:
        plt.savefig(outname+'.pdf')
    return axs

def compare_4vec(predicts, truths, labels=None, nbins=35, min_x=-0.5, max_x=3, **kwargs):
    hist_config = {
        "alpha": 0.8,
        "lw": 2,
        'histtype': 'step',
    }
    axs = view_particle_4vec(predicts, label='prediction', labels=labels, bins=nbins, range=(min_x, max_x), **hist_config, **kwargs)
    view_particle_4vec(truths, axs=axs, label='truth', labels=labels, bins=nbins, range=(min_x, max_x), **hist_config, **kwargs)

def load_yaml(filename):
    with open(filename) as f:
        return yaml.load(f, Loader=yaml.FullLoader)