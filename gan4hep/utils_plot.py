import matplotlib.pyplot as plt
import numpy as np

def add_mean_std(array, x, y, ax, color='k', dy=None, digits=2, fontsize=12, with_std=True):
    this_mean, this_std = np.mean(array), np.std(array)
    if dy is None:
        dy = y * 0.1
    ax.text(x, y, "mean: {0:.{1}f}".format(this_mean, digits), color=color, fontsize=12)
    if with_std:
        ax.text(x, y-dy, "std: {0:.{1}f}".format(this_std, digits), color=color, fontsize=12)
    return ax
        
        
def array2hist(array, ax=None, with_mean_std=True, bins=100, *args, **kwargs):
    if ax is None:
        _, ax = plt.subplots(figsize=(6,6))
    entries, _, _ = ax.hist(array, bins=bins)
    if with_mean_std:
        min_x, max_x = np.min(array), np.max(array)
        x = 0.5*(max_x + min_x)
        y = np.max(entries)*0.8
        add_mean_std(array, x, y, ax)
    return ax


def view_particle_4vec(particles, labels=None, outname=None, *args, **kwargs):
    """
    make histograms of the 4 vectors of the particles
    Inputs:
        particles: 2D arrays (num_particles x 4vectors)
        labels: naming of the 4vectors, default ['E', '$P_x$ [GeV]', '$P_y$ [GeV]', '$P_z$ [GeV]']
    Return:
        ax
    """
    if labels is None:
        labels = ['E', '$P_x$ [GeV]', '$P_y$ [GeV]', '$P_z$ [GeV]']
    fig, axs = plt.subplots(2,2, figsize=(10,10))
    axs = axs.flatten()
    
    for idx in range(4):
        array2hist(particles[:, idx], axs[idx], **kwargs)
        axs[idx].set_xlabel(labels[idx])

    if outname is not None:
        plt.savefig(outname+'.pdf')