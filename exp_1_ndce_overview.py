"""
Check different learning rates and batch sizes for all benchmarks - NDCE. 
"""
from torch.utils.data import DataLoader, TensorDataset
from torch.nn import Sequential, Linear, ReLU, Softmax
from tqdm import tqdm
from torch.optim import Adam
from methods import NonDecisiveCrossEntropyLoss
from torchmetrics import ConfusionMatrix
import numpy as np
import torch
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import balanced_accuracy_score
import math
import os 
import numpy as np

torch.manual_seed(0)
datasets_dir = "datasets"
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('mps')

batch_sizes = [16, 32, 64] 
learning_rates = [1e-3, 1e-4, 1e-5, 1e-6] 
epochs = 150
weights = [0, 0.1, 0.2, 0.3, 0.4]

# Cross validation parameters
n_splits = 2
n_repeats = 5
rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)

datasets_list = os.listdir(datasets_dir)
datasets = []

for data_dir in datasets_list:
    data_path = os.path.join(datasets_dir, data_dir)
    dataset = np.genfromtxt(data_path, delimiter=',')
    datasets.append(dataset)

for batch_idx, batch_size in enumerate(batch_sizes):
    for weight_idx, weight in enumerate(weights):
        for data_idx, data in enumerate(datasets):
            # LOAD DATASET 
            name = datasets_list[data_idx].split(".")[0]

            X_full = data[:,:-1]
            y_full = data[:,-1]

            if X_full.shape[0] % 2 !=0:
                X_full = X_full[1:]
                y_full = y_full[1:]

            # PREPARE SCORES ARRAYS
            n_batches = math.ceil(X_full.shape[0] / n_splits / batch_size)
            
            # learning rates x folds x epochs
            scores_train = np.zeros((len(learning_rates), n_splits * n_repeats, epochs * n_batches))
            scores_test = np.zeros((len(learning_rates), n_splits * n_repeats, epochs))

            # learning rates x folds x epochs
            loss_full = np.zeros((len(learning_rates), n_splits * n_repeats, epochs * n_batches))

            # learning rates x folds x epochs x probas
            probas_train = np.zeros((len(learning_rates), n_splits * n_repeats,
                                    epochs * int(X_full.shape[0]/n_splits), 2))
            probas_test_full = np.zeros((len(learning_rates), n_splits * n_repeats,
                                        epochs, int(X_full.shape[0]/n_splits), 2))
            
            for lr_idx, lr in enumerate(learning_rates):
                for fold, (train, test) in enumerate(rskf.split(X_full, y_full)):
                    print(f"# BS {batch_size} -- WEIGHT {weight} -- DATASET {name} -- LR {lr} -- FOLD {fold}")
                    
                    # CROSS VALIDATION
                    X_train = torch.Tensor(X_full[train])
                    y_train = torch.Tensor(y_full[train]).long()
                    train_dataset = TensorDataset(X_train, y_train)

                    X_test = torch.Tensor(X_full[test])
                    y_test = torch.Tensor(y_full[test]).long()
                    test_dataset = TensorDataset(X_test, y_test)

                    data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

                    model = Sequential(
                                Linear(data_loader.dataset[0][0].shape[0], 1000),
                                ReLU(),
                                Linear(1000, 1000),
                                ReLU(),
                                Linear(1000, 2)
                                ).to(device)

                    optimizer = Adam(model.parameters(), lr=lr)
                    loss_fn = NonDecisiveCrossEntropyLoss(c=weight, w=1-weight)

                    matcal = ConfusionMatrix(num_classes=2, task='multiclass')
                    sofmax = Softmax(dim=1)

                    confmats = []
                    losses = []
                    probas_chunk_epoch = []

                    decisive_acc = []
                    correct_decisive_acc = []
                    accuracies = []

                    for epoch in tqdm(range(epochs)):
                        # print('## EPOCH %i' % epoch)
                        for chunk_id, chunk in enumerate(data_loader):
                            X, y = chunk
                            
                            # Compute prediction and loss
                            X_pm = model(X.to(device))
                            loss = loss_fn(X_pm, y.to(device))
                            
                            losses.append(loss.item())
                            
                            # # Calculate confusion matrix
                            preds = X_pm.argmax(1).to('cpu')
                            
                            # Establish everything on train
                            probas = sofmax(X_pm).detach().cpu().numpy().T
                            decisive_support = sofmax(X_pm).detach().cpu().numpy().max(axis=1)
                            correct_decisive_support = sofmax(X_pm[preds == y]).detach().cpu().numpy().max(axis=1)
                            accuracy = balanced_accuracy_score(y, preds)
                            
                            # Store everything
                            decisive_acc.append(decisive_support)
                            correct_decisive_acc.append(correct_decisive_support)
                            accuracies.append(accuracy)
                            probas_chunk_epoch.append(probas)
                            
                            # Backpropagation
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()

                        # Establish everything on test 
                        X_test_pm = model(X_test.to(device))
                        test_preds = X_test_pm.argmax(1).to('cpu')
                        probas_test = sofmax(X_test_pm).detach().cpu().numpy()
                        
                        # Store everything
                        probas_test_full[lr_idx, fold, epoch] = probas_test
                        scores_test[lr_idx, fold, epoch] = balanced_accuracy_score(y_test, test_preds)
                        
                    loss_full[lr_idx, fold] = np.array(losses)
                    scores_train[lr_idx, fold] = np.array(accuracies)
                    probas_train[lr_idx, fold] = np.array(np.concatenate(probas_chunk_epoch, axis=1).T)

                np.save(f"scores/exp_1/loss_{name}_0_{batch_size}", loss_full)
                np.save(f"scores/exp_1/scores_train_{name}_0_{batch_size}", scores_train)
                np.save(f"scores/exp_1/scores_test_{name}_0_{batch_size}", scores_test)
                np.save(f"scores/exp_1/probas_train_{name}_0_{batch_size}", probas_train)
                np.save(f"scores/exp_1/probas_test_{name}_0_{batch_size}", probas_test_full)
        