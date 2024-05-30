import k3d
import numpy as np
import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from skimage.transform import resize

from utils.dataset import crop_volumn, pad_to_longest_side, resize_volume
from utils.gradcam import GradCAM


def load_sample(lung_path, dose_path, desired_size=(128, 128, 64)):
    lung = np.load(lung_path)
    dose = np.load(dose_path)

    lung, dose = crop_volumn(lung, dose)
    lung, dose = pad_to_longest_side(lung, dose)

    if np.max(dose) - np.min(dose) != 0:
        dose = (dose - np.min(dose)) / (np.max(dose) - np.min(dose))
    else:
        raise Exception("Zero dose !!")

    lung, dose = resize_volume(lung, dose, desired_shape=desired_size)
    return lung, dose


def visualize_slice(lung, dose, n_slices=8):
    interval = lung.shape[2] // n_slices

    slice_indices = [i * interval for i in range(n_slices)]
    fig = plt.figure(figsize=(20, 6))

    count = 1
    for i, idx in enumerate(slice_indices):
        ax = plt.subplot(2, n_slices, count)
        plt.imshow(lung[:, :, idx], cmap="gray")
        plt.title(f"Slice {idx}")
        count += 1
        plt.xticks([])
        plt.yticks([])
    for i, idx in enumerate(slice_indices):
        ax = plt.subplot(2, n_slices, count)
        plt.imshow(dose[:, :, idx], cmap="gray")
        count += 1
        plt.xticks([])
        plt.yticks([])

    plt.tight_layout()
    plt.show()


def visualize_3d(
    lung_path,
    dose_path,
    ignore_threshold=5,
    high_rad_threshold=128,
    low_rad_opacity=0.2,
    high_red_opacity=0.7,
):

    lung1 = np.load(lung_path)
    dose1 = np.load(dose_path)

    lung1_cropped, dose1_cropped = crop_volumn(lung1, dose1)
    lung1_cropped, dose1_cropped = pad_to_longest_side(lung1_cropped, dose1_cropped)

    np.random.seed(0)
    radiation_data = lung1_cropped * dose1_cropped

    max_val = np.max(radiation_data)
    radiation_indices = np.floor(radiation_data / max_val * 254 + 1).astype(np.uint8)

    def rgb_to_hex(r, g, b):
        """Convert an RGB color to a single integer hex value."""
        return (r << 16) + (g << 8) + b

    color_map = [
        (int(255 * i / 254), 0, int(255 * (254 - i) / 254)) for i in range(255)
    ]
    color_map = [rgb_to_hex(r, g, b) for r, g, b in color_map]

    plot = k3d.plot()

    ignored_mask = radiation_indices < ignore_threshold
    low_rad = radiation_indices * (radiation_indices < high_rad_threshold).astype(
        np.uint8
    )
    high_rad = radiation_indices * (radiation_indices >= high_rad_threshold).astype(
        np.uint8
    )

    low_rad[ignored_mask] = 0
    high_rad[ignored_mask] = 0

    plot += k3d.voxels(
        voxels=low_rad,
        color_map=color_map,
        compression_level=9,
        opacity=low_rad_opacity,
        outlines=False,
    )
    plot += k3d.voxels(
        voxels=high_rad,
        color_map=color_map,
        compression_level=9,
        opacity=high_red_opacity,
        outlines=False,
    )

    plot.display()


def run_gradcam(model, lung, dose):
    lung_tensor = torch.from_numpy(lung.transpose((2, 0, 1))).float().unsqueeze(0)
    dose_tensor = torch.from_numpy(dose.transpose((2, 0, 1))).float().unsqueeze(0)
    inputs = lung_tensor * dose_tensor

    heatmaps = []
    for target_layer in [model.conv4, model.conv3]:
        grad_cam = GradCAM(model, target_layer)
        cam = grad_cam.generate_cam(inputs.unsqueeze(0), target_class=0)
        cam = resize(cam.numpy()[0], (64, 128, 128))
        hmin, hmax = np.min(cam), np.max(cam)
        heatmap_normalized = (cam - hmin) / (hmax - hmin)
        heatmaps.append(heatmap_normalized[31])
    return tuple(heatmaps)


def colorize_heatmap(heatmap):
    return plt.cm.jet(np.uint8(heatmap * 255))[:, :, :3]


def gradcam_overlay(heatmap, lung, dose, alpha=0.6):
    return heatmap * (1 - alpha) + alpha * np.repeat(
        (dose[:, :, 32] * lung[:, :, 32])[:, :, np.newaxis], 3, axis=2
    )
