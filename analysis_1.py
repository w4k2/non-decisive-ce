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
batch_sizes = [16] 
learning_rates = [1e-3, 1e-4, 1e-5, 1e-6] 
epochs = 150
colors = ["green", "blue", "purple"]
    
for data_idx, dataset in enumerate(datasets_list):
    fig, ax = plt.subplots(3, 1, figsize=(10, 13))
    for batch_idx, batch_size in enumerate(batch_sizes):
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
        scores_train_ndce = np.load(f"scores/exp_1/scores_train_{name}_0_{batch_size}.npy")
        scores_test_ndce = np.load(f"scores/exp_1/scores_test_{name}_0_{batch_size}.npy")

        scores_train_ce = np.load(f"scores_ce/exp_1/scores_train_{name}_{batch_size}.npy")
        scores_test_ce = np.load(f"scores_ce/exp_1/scores_test_{name}_{batch_size}.npy")

        # datasets x folds x epochs
        loss_ndce = np.load(f"scores/exp_1/loss_{name}_0_{batch_size}.npy")
        loss_ce = np.load(f"scores_ce/exp_1/loss_{name}_{batch_size}.npy") 

        n_batches = math.ceil(X_full.shape[0] / n_splits / batch_size)

        # LOSS AND SCORES
        ndce_epochs = np.zeros((len(learning_rates), n_splits*n_repeats, epochs))
        ndce_scores_epochs = np.zeros((len(learning_rates), n_splits*n_repeats, epochs))
        
        # FILL ZEROS AND DECODE EPOCHS-BATCHES FOR NDCE
        for fold in range(n_splits*n_repeats):
            for lr_idx, lr in enumerate(learning_rates):
                for e in range(epochs):
                    ndce_epochs[lr_idx, fold, e] = np.nansum(loss_ndce[lr_idx, fold, e*n_batches:e*n_batches+n_batches])
                    ndce_scores_epochs[lr_idx, fold, e] = np.mean(scores_train_ndce[lr_idx, fold, e*n_batches:e*n_batches+n_batches])

        ndce_epochs = np.mean(ndce_epochs, axis=1)
        ndce_scores_epochs = np.mean(ndce_scores_epochs, axis=1)
        
        ce_epochs = np.zeros((len(learning_rates), n_splits*n_repeats, epochs))
        ce_scores_epochs = np.zeros((len(learning_rates), n_splits*n_repeats, epochs))

        for fold in range(n_splits*n_repeats):
            for lr_idx, lr in enumerate(learning_rates):
                for e in range(epochs):
                    ce_epochs[lr_idx, fold, e] = np.nansum(loss_ce[lr_idx, fold, e*n_batches:e*n_batches+n_batches])
                    ce_scores_epochs[lr_idx, fold, e] = np.mean(scores_train_ce[lr_idx, fold, e*n_batches:e*n_batches+n_batches])

        # Decode epochs-batches for CE
        ce_epochs = np.mean(ce_epochs, axis=1)
        ce_scores_epochs = np.mean(ce_scores_epochs, axis=1)
        
        ax[0].plot(np.clip(ndce_epochs[0], 1e-5, 1e2), color=colors[batch_idx])
        ax[0].plot(np.clip(ndce_epochs[1], 1e-5, 1e2), color=colors[batch_idx], linestyle="--")
        ax[0].plot(np.clip(ndce_epochs[2], 1e-5, 1e2), color=colors[batch_idx], linestyle=":")
        ax[0].plot(np.clip(ndce_epochs[3], 1e-5, 1e2), color=colors[batch_idx], linestyle="dashdot")
        ax[0].plot(np.clip(ce_epochs[0], 1e-5, 1e2), color="red")
        ax[0].plot(np.clip(ce_epochs[1], 1e-5, 1e2), color="red", linestyle="--")
        ax[0].plot(np.clip(ce_epochs[2], 1e-5, 1e2), color="red", linestyle=":")
        ax[0].plot(np.clip(ce_epochs[3], 1e-5, 1e2), color="red", linestyle="dashdot")
        # ax[0].set_xscale("log")
        #ax[0].set_yscale("log")
        ax[0].grid(ls=":")
        #ax[0].set_xlim(0, 150)
        ax[0].set_title("Loss")
        ax[0].set_xlabel("epochs")
        ax[0].set_ylabel("loss")

        ax[1].plot(ndce_scores_epochs[0], color=colors[batch_idx])
        ax[1].plot(ndce_scores_epochs[1], color=colors[batch_idx], linestyle="--")
        ax[1].plot(ndce_scores_epochs[2], color=colors[batch_idx], linestyle=":")
        ax[1].plot(ndce_scores_epochs[3], color=colors[batch_idx], linestyle="dashdot")
        ax[1].plot(ce_scores_epochs[0], color="red")
        ax[1].plot(ce_scores_epochs[1], color="red", linestyle="--")
        ax[1].plot(ce_scores_epochs[2], color="red", linestyle=":")
        ax[1].plot(ce_scores_epochs[3], color="red", linestyle="dashdot")
        ax[1].grid(ls=":")
        #ax[1].set_xlim(0, 150)
        ax[1].set_title("Train scores")
        ax[1].set_xlabel("epochs")
        ax[1].set_ylabel("balanced accuracy")

        mean_scores_test_ndce = np.mean(scores_test_ndce, axis=1)
        ax[2].plot(mean_scores_test_ndce[0], color=colors[batch_idx])
        ax[2].plot(mean_scores_test_ndce[1], color=colors[batch_idx], linestyle="--")
        ax[2].plot(mean_scores_test_ndce[2], color=colors[batch_idx], linestyle=":")
        ax[2].plot(mean_scores_test_ndce[3], color=colors[batch_idx], linestyle="dashdot")

        mean_scores_test_ce = np.mean(scores_test_ce, axis=1)
        ax[2].plot(mean_scores_test_ce[0], color="red")
        ax[2].plot(mean_scores_test_ce[1], color="red", linestyle="--")
        ax[2].plot(mean_scores_test_ce[2], color="red", linestyle=":")
        ax[2].plot(mean_scores_test_ce[3], color="red", linestyle="dashdot")
        ax[2].grid(ls=":")
        #ax[2].set_xlim(0, 150)
        ax[2].set_title("Test scores")
        ax[2].set_xlabel("epochs")
        ax[2].set_ylabel("balanced accuracy")

    for aa in ax:
        aa.spines['top'].set_visible(False)
        aa.spines['right'].set_visible(False)

    fig.suptitle(f"Dataset {name}", fontsize=16)
    plt.legend(["1e-3", "1e-4", "1e-5", "1e-6"])
    plt.tight_layout()
    # plt.savefig(f"foo.png")
    plt.savefig(f"figures/exp_1/lr_datasets/{name}.png")
    plt.savefig(f"figures/exp_1/lr_datasets/{name}.eps")
    plt.close()

    