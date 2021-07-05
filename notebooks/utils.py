import matplot.pyplot as plt
from gan4hep import gnn_gnn_gan as toGan
from gan4hep.gan_base import GANOptimizer
from gan4hep.graph import loop_dataset
from gan4hep.graph import read_dataset


def load_gnn_GAN(noise_dim, batch_size, ckpt_dir):
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


def plot_ratio(tot, sel, label_tot, label_sel,
                    xlabel, title, **plot_options):
                    
    from more_itertools import pairwise
#     plt.clf()
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(5, 6), sharex=True, gridspec_kw={'height_ratios':[4, 1]})
    fig.subplots_adjust(hspace=0)

    val_tot, bins, _ = ax0.hist(tot, label=label_tot, **plot_options)
    val_sel, bins, _ = ax0.hist(sel, label=label_sel, **plot_options)
    ax0.legend(fontsize=14, loc='upper right')
    ax0.set_title(title)

    ratio = [x/y if y != 0 else 0. for x,y in zip(val_sel, val_tot)][:-1]
    xvals = [0.5*(x[0]+x[1]) for x in pairwise(bins)][1:]
    ax1.plot(xvals, ratio, 'o-', label='ratio', lw=2, markersize=5)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel('ratio')
    ax1.set_ylim(-5.2, 5.2)
    
    
def visiual(predicts, truths, ngen,
            nbins=35, min_x=-0.5, max_x=3, use_pt_eta_phi_E=False):    
    hist_config = {
        "alpha": 0.8,
        "lw": 2,
        'histtype': 'step',
    }

    config_4vector = [
        dict([("bins",nbins), ("range",(min_x, max_x))]),
        dict([("bins",nbins), ("range",(min_x, max_x))]),
        dict([("bins",nbins), ("range",(min_x, max_x))]),
        dict([("bins",nbins), ("range",(min_x, max_x))])
    ]
    
    if use_pt_eta_phi_E:
        xlabels = ['pT [GeV]', 'eta [GeV]', 'phi [GeV]', 'E [GeV]']
    else:
        xlabels = ['px [GeV]', 'py [GeV]', 'pz [GeV]', 'E [GeV]']

    xp = [2]*4
    yp = np.array([500] + [250]*3)
    dy = np.array([50] + [25]*3)
    
    def plot_4vector(offset):
        _, axs = plt.subplots(2,2, figsize=(10,10), constrained_layout=True)
        axs = axs.flatten()
        for ix in range(4):
            idx = ix
            axs[ix].hist(predicts[:, offset, idx], **hist_config, **config_4vector[ix], label="prediction")
            axs[ix].hist(truths[:, offset, idx], **hist_config, **config_4vector[ix], label="truth")
            axs[ix].set_xlabel(xlabels[ix])
        axs[ix].legend(loc='upper right')
            
    for idx in range(1,3):
        plot_4vector(idx)
        
        
