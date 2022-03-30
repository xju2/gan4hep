# %% 
import matplotlib.pyplot as plt
import awesome_plot as ap

import numpy as np

import tensorflow as tf
from tensorflow.core.util import event_pb2
# %%
event_file = "/media/DataOcean/projects/ml/herwig/ClusterDecayer/TrainedWithConvertedInputs/logs/20220222-144205/events.out.tfevents.1645569725.Faraday.459425.0.v2"
summary_dict = {
    "tot_wasserstein_dis": [],
    "G_loss": [],
    "D_loss": []
}
key_values = [k for k in summary_dict.keys()]
tot_wdis = []
for raw_record in tf.data.TFRecordDataset(event_file):
    for value in event_pb2.Event.FromString(raw_record.numpy()).summary.value:
            # print("value: {!r} ;".format(value))
            if value.tag in key_values:
                if value.tensor.ByteSize():
                    t = tf.make_ndarray(value.tensor)
                    summary_dict[value.tag].append(t)
# %% 
def get_smallest(a_list: np.ndarray):
    """a_list: [1, 2, 4]"""
    min_val = np.max(a_list)
    new_list = []
    for x in a_list:
        if x < min_val:
            new_list.append(x)
            min_val = x
        else:
            new_list.append(min_val)
    return new_list
# %%
ap.set_config()
best_wdis = get_smallest(summary_dict["tot_wasserstein_dis"])

fig, ax = plt.subplots(figsize=(6,6))
ax.plot(np.array(summary_dict["D_loss"])/2., lw=2, label='Discriminator Loss / 2')
ax.plot(np.array(summary_dict["G_loss"]), lw=2, label="Generator Loss")
ax.set_xlabel("Epochs")
ax.set_ylabel("Losses")
ax.tick_params(width=2, grid_alpha=0.5, labelsize=12)
ax.legend(fontsize=14, frameon=False)
ax.grid(True, axis='y')

ax2 = ax.twinx()
color='red'
ax2.set_ylabel("Best Wasserstein Dis", color=color, fontsize=14)
ax2.tick_params(width=2, grid_alpha=0.5, labelsize=12, axis='y', labelcolor=color)
# ax2.plot(summary_dict["tot_wasserstein_dis"], color=color, label='Total Wasserstein Dis.')
ax2.plot(best_wdis, color=color, lw=2, label="Best Wasserstein Dis.")
# ax2.grid(True, axis='y')
fig.tight_layout()
plt.savefig("loss_vs_epochs.pdf")
# %%

