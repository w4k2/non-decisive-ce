"""
First experiments (benchmarks) results - figures. 
"""
import numpy as np
import matplotlib.pyplot as plt
import math
import os

datasets_dir = "datasets"
datasets_list = os.listdir(datasets_dir)

batch_size = 16
epochs = 150
n_splits = 2
n_repeats = 5

for data_idx, dataset in enumerate(datasets_list):
    name = dataset.split(".")[0]
    data_path = os.path.join(datasets_dir, dataset)
    data = np.genfromtxt(data_path, delimiter=',')

    X_full = data[:,:-1]
    y_full = data[:,-1]

    if X_full.shape[0] % 2 !=0:
        X_full = X_full[1:]
        y_full = y_full[1:]

    n_batches = math.ceil(X_full.shape[0] / n_splits / batch_size)

    # datasets x folds x epochs
    try: 
        print(f"LOADING {name}")
        # NORMAL
        scores_train_ndce_normal = np.load(f"scores/exp_1/scores_train_{name}_normal.npy")
        scores_test_ndce_normal = np.load(f"scores/exp_1/scores_test_{name}_normal.npy")

        scores_train_ce_normal = np.load(f"scores_ce/exp_1/scores_train_{name}_normal.npy")
        scores_test_ce_normal = np.load(f"scores_ce/exp_1/scores_test_{name}_normal.npy")

        # datasets x folds x epochs
        loss_ndce_normal = np.load(f"scores/exp_1/loss_{name}_normal.npy")
        loss_ce_normal = np.load(f"scores_ce/exp_1/loss_{name}_normal.npy")

        # RAW
        scores_train_ndce_raw = np.load(f"scores/exp_1/scores_train_{name}_raw.npy")
        scores_test_ndce_raw = np.load(f"scores/exp_1/scores_test_{name}_raw.npy")

        scores_train_ce_raw = np.load(f"scores_ce/exp_1/scores_train_{name}_raw.npy")
        scores_test_ce_raw = np.load(f"scores_ce/exp_1/scores_test_{name}_raw.npy")

        # datasets x folds x epochs
        loss_ndce_raw = np.load(f"scores/exp_1/loss_{name}_raw.npy")
        loss_ce_raw = np.load(f"scores_ce/exp_1/loss_{name}_raw.npy")

        n_batches = math.ceil(X_full.shape[0] / n_splits / batch_size)

        # LOSS AND SCORES - NORMAL
        ndce_epochs_normal = np.zeros((n_splits*n_repeats, epochs))
        ndce_scores_epochs_normal = np.zeros((n_splits*n_repeats, epochs))
        
        # FILL ZEROS AND DECODE EPOCHS-BATCHES FOR NDCE
        for fold in range(n_splits*n_repeats):
            for e in range(epochs):
                ndce_epochs_normal[fold, e] = np.nansum(loss_ndce_normal[fold, e*n_batches:e*n_batches+n_batches])
                ndce_scores_epochs_normal[fold, e] = np.mean(scores_train_ndce_normal[fold, e*n_batches:e*n_batches+n_batches])

        ndce_epochs_normal = np.mean(ndce_epochs_normal, axis=0)
        ndce_scores_epochs_normal = np.mean(ndce_scores_epochs_normal, axis=0)

        # Decode epochs-batches for CE
        mean_ce_normal = np.mean(loss_ce_normal, axis=0)
        mean_ce_scores_normal = np.mean(scores_train_ce_normal, axis=0)
        
        ce_epochs_normal = np.zeros((epochs))
        ce_scores_epochs_normal = np.zeros((epochs))
        for e in range(epochs):
            ce_epochs_normal[e] = np.nansum(mean_ce_normal[e*n_batches:e*n_batches+n_batches])  
            ce_scores_epochs_normal[e] = np.mean(mean_ce_scores_normal[e*n_batches:e*n_batches+n_batches])

        # LOSS AND SCORES - RAW
        ndce_epochs_raw = np.zeros((n_splits*n_repeats, epochs))
        ndce_scores_epochs_raw = np.zeros((n_splits*n_repeats, epochs))
        
        # FILL ZEROS AND DECODE EPOCHS-BATCHES FOR NDCE
        for fold in range(n_splits*n_repeats):
            for e in range(epochs):
                ndce_epochs_raw[fold, e] = np.nansum(loss_ndce_raw[fold, e*n_batches:e*n_batches+n_batches])
                ndce_scores_epochs_raw[fold, e] = np.mean(scores_train_ndce_raw[fold, e*n_batches:e*n_batches+n_batches])

        ndce_epochs_raw = np.mean(ndce_epochs_raw, axis=0)
        ndce_scores_epochs_raw = np.mean(ndce_scores_epochs_raw, axis=0)

        # Decode epochs-batches for CE
        mean_ce_raw = np.mean(loss_ce_raw, axis=0)
        mean_ce_scores_raw = np.mean(scores_train_ce_raw, axis=0)
        
        ce_epochs_raw = np.zeros((epochs))
        ce_scores_epochs_raw = np.zeros((epochs))
        for e in range(epochs):
            ce_epochs_raw[e] = np.nansum(mean_ce_raw[e*n_batches:e*n_batches+n_batches])  
            ce_scores_epochs_raw[e] = np.mean(mean_ce_scores_raw[e*n_batches:e*n_batches+n_batches])

        fig, ax = plt.subplots(3, 1, figsize=(10, 13))
        ax[0].plot(np.clip(ndce_epochs_normal, 1e-5, 1e2), color="green")
        ax[0].plot(np.clip(ce_epochs_normal, 1e-5, 1e2), color="red")

        ax[0].plot(np.clip(ndce_epochs_raw, 1e-5, 1e2), color="green", linestyle="--")
        ax[0].plot(np.clip(ce_epochs_raw, 1e-5, 1e2), color="red", linestyle="--")
        # ax[0].set_xscale("log")
        #ax[0].set_yscale("log")
        ax[0].grid(ls=":")
        #ax[0].set_xlim(0, 150)
        ax[0].set_title("Loss")
        ax[0].set_xlabel("epochs")
        ax[0].set_ylabel("loss")

        ax[1].plot(ndce_scores_epochs_normal, color="green")
        ax[1].plot(ce_scores_epochs_normal, color="red")

        ax[1].plot(ndce_scores_epochs_raw, color="green", linestyle="--")
        ax[1].plot(ce_scores_epochs_raw, color="red", linestyle="--")
        ax[1].grid(ls=":")
        #ax[1].set_xlim(0, 150)
        ax[1].set_title("Train scores")
        ax[1].set_xlabel("epochs")
        ax[1].set_ylabel("balanced accuracy")

        ax[2].plot(np.mean(scores_test_ndce_normal, axis=0), color="green")
        ax[2].plot(np.mean(scores_test_ce_normal, axis=0), color="red")

        ax[2].plot(np.mean(scores_test_ndce_raw, axis=0), color="green", linestyle="--")
        ax[2].plot(np.mean(scores_test_ce_raw, axis=0), color="red", linestyle="--")
        ax[2].grid(ls=":")
        #ax[2].set_xlim(0, 150)
        ax[2].set_title("Test scores")
        ax[2].set_xlabel("epochs")
        ax[2].set_ylabel("balanced accuracy")

        fig.suptitle(f"Dataset {name}", fontsize=16)
        plt.tight_layout()
        plt.savefig(f"figures/exp_1/loss-scores-{name}.png")
        plt.close()
    except FileNotFoundError:
        print("NOT YET")
    