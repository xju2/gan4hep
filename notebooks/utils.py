import matplot.pyplot as plt

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