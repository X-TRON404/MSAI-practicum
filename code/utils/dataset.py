import torch
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import ndimage
import random
from torch.utils.data import Dataset
from torch.utils.data.sampler import BatchSampler, RandomSampler


def crop_mask_z_axis(mask):
    # crop z-axis
    # example:
    #   [0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0] =>
    #   [1, 1, 1, 1, 0, 1]
    nz_mask = np.nonzero(mask)
    min_z, max_z = nz_mask[2].min(), nz_mask[2].max()
    return mask[:, :, min_z : max_z + 1], (min_z, max_z)


def crop_volumn(mask, dose):
    nz_mask = np.nonzero(mask)
    min_x, max_x = nz_mask[0].min(), nz_mask[0].max()
    min_y, max_y = nz_mask[1].min(), nz_mask[1].max()
    min_z, max_z = nz_mask[2].min(), nz_mask[2].max()
    return (
        mask[min_x : max_x + 1, min_y : max_y + 1, min_z : max_z + 1],
        dose[min_x : max_x + 1, min_y : max_y + 1, min_z : max_z + 1],
    )


def pad_to_longest_side(lung, dose):
    H, W, D = lung.shape
    max_length = max(H, W)

    padded_lung = np.zeros((max_length, max_length, D), dtype=lung.dtype)
    padded_dose = np.zeros((max_length, max_length, D), dtype=dose.dtype)

    h_offset = (max_length - H) // 2
    w_offset = (max_length - W) // 2

    padded_lung[h_offset : h_offset + H, w_offset : w_offset + W, :] = lung
    padded_dose[h_offset : h_offset + H, w_offset : w_offset + W, :] = dose

    return padded_lung, padded_dose


def resize_volume(lung, dose, desired_shape=(128, 128, 64)):
    # resize volumn to desired shape
    dh, dw, dz = desired_shape
    ch, cw, cz = lung.shape
    depth = cz / dz
    width = cw / dw
    height = ch / dh
    depth_factor = 1 / depth
    width_factor = 1 / width
    height_factor = 1 / height

    lung = ndimage.zoom(
        lung, (height_factor, width_factor, depth_factor), order=0
    )  # nearst
    dose = ndimage.zoom(
        dose, (height_factor, width_factor, depth_factor), order=1
    )  # bilinear
    return lung, dose


def process_data(
    lung_path,
    dose_path,
    label_path,
    downsample_neg=False,
    testing=False,
    npos=48,
    nneg=48,
    eqd2=False,
    ab=3,
):
    df = pd.read_csv(label_path)

    lung_fnames = sorted(os.listdir(lung_path))
    dose_fnames = sorted(os.listdir(dose_path))

    lung_prefix = [fname.strip("_l.npy") for fname in lung_fnames]
    dose_prefix = [fname.strip("_d.npy") for fname in dose_fnames]

    lung_dose_intersection = sorted(
        list(set(lung_prefix).intersection(set(dose_prefix)))
    )
    lung_dose_label_intersection = sorted(
        list(set(df["anon_id"].to_list()).intersection(list(lung_dose_intersection)))
    )

    labels_raw = df[["anon_id", "pneumonitis", "fractions"]].to_dict()
    labels_id = []
    labels = []
    fractions = []
    n_pos, n_neg = 0, 0

    for i in range(len(list(labels_raw["anon_id"].keys()))):
        if labels_raw["anon_id"][i] in lung_dose_label_intersection:
            if labels_raw["anon_id"][i] in labels_id:
                continue
            labels_id.append(labels_raw["anon_id"][i])
            labels.append(int(labels_raw["pneumonitis"][i]))
            fractions.append(float(labels_raw["fractions"][i]))
            if labels_raw["pneumonitis"][i] == 0:
                n_neg += 1
            else:
                n_pos += 1

    print(f"Total Samples: {len(lung_dose_label_intersection)}")
    print(f"Total Positive Samples: {n_pos}")
    print(f"Total Negative Samples: {n_neg}\n")
    print("loading data ...")
    samples = []
    _npos, _nneg = 0, 0
    for label_id, label, fraction in tqdm(zip(labels_id, labels, fractions)):
        if downsample_neg or testing:
            if _npos >= npos and label == 1:
                continue
            if _nneg >= nneg and label == 0:
                continue

        lung = np.load(os.path.join(lung_path, label_id + "_l.npy"))
        dose = np.load(os.path.join(dose_path, label_id + "_d.npy"))

        try:
            if eqd2:
                dose = dose * (fraction + ab) / (2 + ab)
            lung, dose = crop_volumn(lung, dose)
            lung, dose = pad_to_longest_side(lung, dose)

            if np.max(dose) - np.min(dose) != 0:
                dose = (dose - np.min(dose)) / (np.max(dose) - np.min(dose))
            else:
                print(f"{label_id} with label {label} with zero-dose !!")

            lung, dose = resize_volume(lung, dose)
            samples.append((label_id, lung, dose, label))

            if downsample_neg or testing:
                if label == 0:
                    _nneg += 1
                else:
                    _npos += 1

        except Exception as e:
            print(e)
            print(f"{label_id} with label {label} skipped !!")

    print("... finished\n")
    return samples


def split_train_test(samples, ratio=0.8):
    pos, neg = [], []
    for sample in samples:
        label_id, lung, dose, label = sample
        if label == 1:
            pos.append(sample)
        else:
            neg.append(sample)
    n_pos_train = int(len(pos) * ratio)
    n_neg_train = int(len(neg) * ratio)

    train_samples = pos[:n_pos_train] + neg[:n_neg_train]
    test_samples = pos[n_pos_train:] + neg[n_neg_train:]

    random.shuffle(train_samples)
    random.shuffle(test_samples)

    return train_samples, test_samples


class Pneumonitis(Dataset):
    def __init__(self, data, aug=False, return_id=False) -> None:
        super().__init__()
        self.aug = aug
        self.return_id = return_id
        self.data = data

    def __getitem__(self, index):
        label_id, lung, dose, label = self.data[index]
        lung = torch.from_numpy(lung.transpose((2, 0, 1))).float().unsqueeze(0)
        dose = torch.from_numpy(dose.transpose((2, 0, 1))).float().unsqueeze(0)
        label = torch.tensor([label]).float()
        if self.aug:
            dose += torch.randn(dose.size()) * 0.1
        if self.return_id:
            return label_id, lung, dose, label
        return lung, dose, label

    def __len__(self):
        return len(self.data)


class PneumonitisDownsample(Dataset):
    def __init__(self, data, aug=False, return_id=False) -> None:
        super().__init__()
        self.aug = aug
        self.return_id = return_id
        self.data = data

    def __getitem__(self, index):
        label_id, lung, dose, label = self.data[index]
        lung = torch.from_numpy(lung.transpose((2, 0, 1))).float().unsqueeze(0)
        dose = torch.from_numpy(dose.transpose((2, 0, 1))).float().unsqueeze(0)
        label = torch.tensor([label]).float()
        if self.aug:
            dose += torch.randn(dose.size()) * 0.1
        if self.return_id:
            return label_id, lung, dose, label

    def __len__(self):
        return len(self.data)


class BalancedBatchPneumonitis(Dataset):
    def __init__(self, data, aug=False) -> None:
        super().__init__()
        self.aug = aug
        self.data = data
        self.positive_indices = [
            i for i, (_, _, _, label) in enumerate(data) if label == 1
        ]
        self.negative_indices = [
            i for i, (_, _, _, label) in enumerate(data) if label == 0
        ]

    def __getitem__(self, index):
        label_id, lung, dose, label = self.data[index]
        lung = torch.from_numpy(lung.transpose((2, 0, 1))).float().unsqueeze(0)
        dose = torch.from_numpy(dose.transpose((2, 0, 1))).float().unsqueeze(0)
        label = torch.tensor([label]).float()
        if self.aug:
            dose += torch.randn(dose.size()) * 0.1
        return lung, dose, label

    def __len__(self):
        return len(self.data)


class BalancedBatchSampler(BatchSampler):
    def __init__(self, positive_indices, negative_indices, batch_size):
        self.positive_indices = positive_indices
        self.negative_indices = negative_indices
        self.batch_size = batch_size
        self.pos_batch_size = self.batch_size // 2  # Assuming you want 50/50 split
        self.neg_batch_size = self.batch_size - self.pos_batch_size

        self.pos_sampler = RandomSampler(self.positive_indices, replacement=True)
        self.neg_sampler = RandomSampler(self.negative_indices)

    def __iter__(self):
        self.pos_iter = iter(self.pos_sampler)
        self.neg_iter = iter(self.neg_sampler)
        while True:
            batch = []
            pos_exhausted = neg_exhausted = False
            for _ in range(self.pos_batch_size):
                try:
                    batch.append(next(self.pos_iter))
                except StopIteration:
                    pos_exhausted = True
                    break
            if not pos_exhausted:
                for _ in range(self.neg_batch_size):
                    try:
                        batch.append(next(self.neg_iter))
                    except StopIteration:
                        neg_exhausted = True
                        break
            if pos_exhausted or neg_exhausted:
                if neg_exhausted:
                    self.neg_iter = iter(self.neg_sampler)
                if len(batch) == self.batch_size:
                    yield batch
                break
            else:
                yield batch

    def __len__(self):
        return min(
            len(self.positive_indices) // self.pos_batch_size,
            len(self.negative_indices) // self.neg_batch_size,
        )
