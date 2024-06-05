import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import os
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam

from models.resnet3d import ResNet, Bottleneck
from models.cnn3d import CNN3D
from models.criterians import FocalLoss
from utils.dataset import (
    split_train_test,
    process_data,
    Pneumonitis,
    BalancedBatchPneumonitis,
    BalancedBatchSampler,
)


def accuracy(pred, label):
    return torch.sum((pred > 0.5).float() == label) / pred.shape[0]


def test(model, test_loader, run_label, save=True, is_train_loader=False):
    preds, labels = [], []
    val_acc, n_items_val = 0, 0
    with torch.no_grad():
        for sample in tqdm(test_loader):
            if is_train_loader:
                lung, dose, label = sample
            else:
                sid, lung, dose, label = sample

            fusion = (lung * dose).cuda()
            label = label.cuda()
            pred = model(fusion)

            val_acc += torch.sum(((pred > 0.5).int() == label).float())
            n_items_val += pred.size(0)

            preds.append(pred.cpu().numpy())
            labels.append(label.cpu().numpy())

    preds = np.concatenate(preds, axis=0)
    labels = np.concatenate(labels, axis=0)

    print(f"Acc - {val_acc / n_items_val}")

    if save:
        np.save(f"./results/predictions/{run_label}-preds.npy", preds)
        np.save(f"./results/predictions/{run_label}-labels.npy", labels)


def get_model(name):
    if name == "cnn":
        return CNN3D()
    elif name == "resnet50":
        return ResNet(Bottleneck, [3, 4, 6, 3])
    else:
        raise Exception(f"model {name} not found")


def main(
    train_samples=None,
    test_samples=None,
    run_label="baseline",
    model_name="resnet",
    epochs=10,
    batchsize=16,
    lr=1e-4,
    test_interval=1,
    criterian_name="bce",
    batch_balence=False,
    aug=False,
):
    save_path = "./results/ckpts"
    os.makedirs(save_path, exist_ok=True)
    os.makedirs("./results/logs", exist_ok=True)
    os.makedirs("./results/predictions", exist_ok=True)
    train_log = open(f"./results/logs/{run_label}-train-log.txt", "w")
    test_log = open(f"./results/logs/{run_label}-test-log.txt", "w")

    if not batch_balence:
        train_set = Pneumonitis(train_samples, aug=aug)
        train_loader = DataLoader(
            train_set, batch_size=batchsize, shuffle=True, num_workers=2
        )
    else:
        train_set = BalancedBatchPneumonitis(train_samples, aug=aug)
        balanced_sampler = BalancedBatchSampler(
            train_set.positive_indices, train_set.negative_indices, batch_size=batchsize
        )
        train_loader = DataLoader(
            train_set, batch_sampler=balanced_sampler, num_workers=2
        )

    train_set_test = Pneumonitis(train_samples, aug=False)
    train_loader_test = DataLoader(train_set_test, batch_size=1, shuffle=False)

    test_set = Pneumonitis(test_samples, return_id=True)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    if criterian_name == "bce":
        criterian = nn.BCELoss()
    elif criterian_name == "focal_loss":
        criterian = FocalLoss(alpha=0.9)

    model = get_model(model_name).cuda()
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=5e-4)

    for epoch in range(epochs):
        model.train()
        train_acc, n_items_tr = 0, 0
        for lung, dose, label in tqdm(train_loader):
            fusion = (lung * dose).cuda()
            label = label.cuda()

            optimizer.zero_grad()

            pred = model(fusion)
            loss = criterian(pred, label)
            loss.backward()
            optimizer.step()

            train_acc += torch.sum(((pred > 0.5).int() == label).float())
            n_items_tr += pred.size(0)

        epoch_train_acc = train_acc / n_items_tr
        print(f"Epoch {epoch + 1}: train-acc - {epoch_train_acc}")
        train_log.write(f"{epoch_train_acc}\n")

        if (epoch + 1) % test_interval == 0:
            model.eval()
            test_acc, n_items_te = 0, 0
            with torch.no_grad():
                for sid, lung, dose, label in tqdm(test_loader):
                    fusion = (lung * dose).cuda()
                    label = label.cuda()

                    pred = model(fusion)
                    test_acc += torch.sum(((pred > 0.5).int() == label).float())
                    n_items_te += pred.size(0)

            epoch_test_acc = test_acc / n_items_te
            print(f"Epoch {epoch + 1}: test-acc - {epoch_test_acc}")
            test_log.write(f"{epoch_test_acc}\n")

    model.eval()
    print("Train-set results")
    test(model, train_loader_test, run_label + "-train", is_train_loader=True)
    print("Test-set results")
    test(model, test_loader, run_label)

    torch.save(
        model.state_dict(), os.path.join(save_path, f"pneumonitis-{run_label}.pt")
    )


if __name__ == "__main__":
    lung_path = "/projects/p32050/lung"
    dose_path = "/projects/p32050/dose"
    label_path = "/projects/p32050/all_stats.csv"

    downsample_neg = True
    eqd2 = False

    print("loading data ...")
    samples = process_data(
        lung_path, dose_path, label_path, downsample_neg=downsample_neg, eqd2=eqd2
    )
    train_samples, test_samples = split_train_test(samples, ratio=0.8)
    print("data loaded")

    main(
        train_samples,
        test_samples,
        run_label="cnn-dn2",
        model_name="cnn",
        epochs=20,
        batchsize=16,
        lr=1e-4,
        test_interval=1,
        criterian_name="bce",
        batch_balence=False,
        aug=False,
    )
