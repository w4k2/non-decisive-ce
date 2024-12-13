"""
Time evaluation for EMNIST and CE. 
"""
from torch.utils.data import DataLoader
import torchvision.transforms as transforms 
from torch.nn import Sequential, Linear, ReLU, Softmax, Conv2d, MaxPool2d, Flatten, CrossEntropyLoss
from tqdm import tqdm
from torch.optim import Adam
from torchmetrics import ConfusionMatrix
import numpy as np
import torch
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import balanced_accuracy_score
import math
import numpy as np
from torchvision.datasets import EMNIST
from torch.utils.data import SubsetRandomSampler
import time

datasets_dir = "datasets"
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('mps')

torch.manual_seed(0)

batch_size = 128
learning_rate = 1e-5
epochs = 150

# Cross validation parameters
n_splits = 2
n_repeats = 5
rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)

dataset = EMNIST(root='data/', split='mnist', transform=transforms.ToTensor(), download=False)

# PREPARE SCORES ARRAYS
n_batches = math.ceil(len(dataset) / n_splits / batch_size)

# folds x epochs
scores_train = np.zeros((n_splits * n_repeats, epochs * n_batches))
scores_test = np.zeros((n_splits * n_repeats, epochs * n_batches))

# times
times_train = np.zeros((n_splits * n_repeats, epochs))
times_test = np.zeros((n_splits * n_repeats, epochs))

# datasets x folds x epochs
loss_full = np.zeros((n_splits * n_repeats, 
                    epochs * math.ceil(len(dataset) / n_splits / batch_size)))

for fold, (train, test) in enumerate(rskf.split(dataset.targets, dataset.targets)):
    train_sampler = SubsetRandomSampler(train)
    test_sampler = SubsetRandomSampler(test)

    train_data = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler)
    test_data = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler)

    model = Sequential(
                Conv2d(1, 32, kernel_size=3, padding=1), 
                MaxPool2d(2),
                Flatten(),
                Linear(6272, 64),
                ReLU(),
                Linear(64, 1024),
                ReLU(),
                Linear(1024, 64),
                ReLU(),
                Linear(64, len(dataset.classes))
                ).to(device)

    optimizer = Adam(model.parameters(), lr=learning_rate)
    loss_fn = CrossEntropyLoss()

    matcal = ConfusionMatrix(num_classes=len(dataset.classes), task='multiclass')
    sofmax = Softmax(dim=1)

    losses = []
    accuracies = []
    test_accuracies = []

    for epoch in tqdm(range(epochs)):
        print('## EPOCH %i' % epoch)
        
        model.train() 
        t_tr = time.time()
        for chunk_id, chunk in enumerate(tqdm(train_data)):
            X, y = chunk
        
            # Compute prediction and loss
            X_pm = model(X.to(device))
            loss = loss_fn(X_pm, y.to(device))
            
            losses.append(loss.item())
            
            # Calculate confusion matrix
            preds = X_pm.argmax(1).to('cpu')
            
            # Establish everything on train
            accuracy = balanced_accuracy_score(y, preds)
            accuracies.append(accuracy)
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
            # old_grads = grads
        times_train[fold, epoch] = time.time() - t_tr

        # Establish everything on test
        model.eval() 
        t_ts = time.time()
        for chunk_id, chunk in enumerate(tqdm(test_data)):
            X, y = chunk

            X_test_pm = model(X.to(device))
            test_preds = X_test_pm.argmax(1).to('cpu')
            test_accuracy = balanced_accuracy_score(y, test_preds)

            test_accuracies.append(test_accuracy)

        times_test[fold, epoch] = time.time() - t_ts

    # Store everything
    loss_full[fold] = np.array(losses)
    scores_train[fold] = np.array(accuracies)
    scores_test[fold] = np.array(test_accuracies)  

    np.save(f"scores_ce/exp_2/loss", loss_full)
    np.save(f"scores_ce/exp_2/scores_train", scores_train)
    np.save(f"scores_ce/exp_2/scores_test", scores_test)
    np.save(f"scores_ce/exp_2/times_train", times_train)
    np.save(f"scores_ce/exp_2/times_test", times_test)
