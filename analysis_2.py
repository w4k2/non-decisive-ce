"""
Second experiments (EMNIST) results - figures. 
"""
import numpy as np
import matplotlib.pyplot as plt
import math
import os

batch_size = 128
epochs = 150
n_splits = 2
n_repeats = 5
len_dataset = 60000
n_batches = math.ceil(len_dataset / n_splits / batch_size)

# folds x epochs
scores_train_ndce = np.load(f"scores/exp_2/scores_train.npy")
scores_test_ndce = np.load(f"scores/exp_2/scores_test.npy")

scores_train_ce = np.load(f"scores_ce/exp_2/scores_train.npy")
scores_test_ce = np.load(f"scores_ce/exp_2/scores_test.npy")

# folds x epochs
loss_ndce = np.load(f"scores/exp_2/loss.npy")
loss_ce = np.load(f"scores_ce/exp_2/loss.npy")

times_train_ndce = np.load(f"scores/exp_2/times_train.npy")
times_test_ndce = np.load(f"scores/exp_2/times_test.npy")

times_train_ce = np.load(f"scores_ce/exp_2/times_train.npy")
times_test_ce = np.load(f"scores_ce/exp_2/times_test.npy")

# LOSS AND SCORES
ndce_epochs = np.zeros((n_splits*n_repeats, epochs))
ndce_scores_epochs = np.zeros((n_splits*n_repeats, epochs))
ndce_scores_epochs_test = np.zeros((n_splits*n_repeats, epochs))

for fold in range(n_splits*n_repeats):
    for e in range(epochs):
        ndce_epochs[fold, e] = np.nansum(loss_ndce[fold, e*n_batches:e*n_batches+n_batches])
        ndce_scores_epochs[fold, e] = np.mean(scores_train_ndce[fold, e*n_batches:e*n_batches+n_batches])
        ndce_scores_epochs_test[fold, e] = np.mean(scores_test_ndce[fold, e*n_batches:e*n_batches+n_batches])


ndce_epochs = np.mean(ndce_epochs, axis=0)
ndce_scores_epochs = np.mean(ndce_scores_epochs, axis=0)
ndce_scores_epochs_test = np.mean(ndce_scores_epochs_test, axis=0)

mean_ce = np.mean(loss_ce, axis=0)
ce_epochs = np.zeros((epochs))

mean_ce_scores = np.mean(scores_train_ce, axis=0)
ce_scores_epochs = np.zeros((epochs))
mean_ce_scores_test = np.mean(scores_test_ce, axis=0)
ce_scores_epochs_test = np.zeros((epochs))

for e in range(epochs):
    ce_epochs[e] = np.nansum(mean_ce[e*n_batches:e*n_batches+n_batches])  
    ce_scores_epochs[e] = np.mean(mean_ce_scores[e*n_batches:e*n_batches+n_batches])
    ce_scores_epochs_test[e] = np.mean(mean_ce_scores_test[e*n_batches:e*n_batches+n_batches])


fig, ax = plt.subplots(3, 1, figsize=(10, 13))
ax[0].plot(np.mean(times_train_ndce, axis=0), color="green")
ax[0].plot(np.mean(times_test_ndce, axis=0), color="green", linestyle=":")
ax[0].plot(np.mean(times_train_ce, axis=0), color="red")
ax[0].plot(np.mean(times_test_ce, axis=0), color="red", linestyle=":")


ax[0].grid(ls=":")

ax[0].set_title("Time")
ax[0].set_xlabel("epochs")
ax[0].set_ylabel("Time [s]")

ax[1].plot(ndce_scores_epochs, color="green")
ax[1].plot(ce_scores_epochs, color="red")
ax[1].grid(ls=":")
#ax[1].set_xlim(0, 150)
ax[1].set_title("Train scores")
ax[1].set_xlabel("epochs")
ax[1].set_ylabel("balanced accuracy")

ax[2].plot(ndce_scores_epochs_test, color="green")
ax[2].plot(ce_scores_epochs_test, color="red")
ax[2].grid(ls=":")
#ax[2].set_xlim(0, 150)
ax[2].set_title("Test scores")
ax[2].set_xlabel("epochs")
ax[2].set_ylabel("balanced accuracy")

for aa in ax:
    aa.spines['top'].set_visible(False)
    aa.spines['right'].set_visible(False)

plt.legend(["NDCE", "CE"])
fig.suptitle(f"Dataset EMNIST", fontsize=16)
plt.tight_layout()
plt.savefig(f"figures/exp_2/time-scores.png")
plt.savefig(f"figures/exp_2/time.eps")
plt.close()