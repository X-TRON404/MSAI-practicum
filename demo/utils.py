import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage.transform import resize

from gradcam import GradCAM


def crop_volumn(mask, dose):
    nz_mask = np.nonzero(mask)
    min_x, max_x = nz_mask[0].min(), nz_mask[0].max()
    min_y, max_y = nz_mask[1].min(), nz_mask[1].max()
    min_z, max_z = nz_mask[2].min(), nz_mask[2].max()
    return mask[min_x:max_x+1, min_y:max_y+1, min_z:max_z+1], \
        dose[min_x:max_x+1, min_y:max_y+1, min_z:max_z+1]

def pad_to_longest_side(lung, dose):
    H, W, D = lung.shape
    max_length = max(H, W)
    
    padded_lung = np.zeros((max_length, max_length, D), dtype=lung.dtype)
    padded_dose = np.zeros((max_length, max_length, D), dtype=dose.dtype)

    h_offset = (max_length - H) // 2
    w_offset = (max_length - W) // 2
    
    padded_lung[h_offset:h_offset+H, w_offset:w_offset+W, :] = lung
    padded_dose[h_offset:h_offset+H, w_offset:w_offset+W, :] = dose
    
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
    
    lung = ndimage.zoom(lung, (height_factor, width_factor, depth_factor), order=0)  # nearst
    dose = ndimage.zoom(dose, (height_factor, width_factor, depth_factor), order=1)  # bilinear
    return lung, dose

def load_sample(lung_path, dose_path, desired_size=(128, 128, 64)):
    lung = np.load(lung_path)
    dose = np.load(dose_path)
    
    lung, dose = crop_volumn(lung, dose)
    lung, dose = pad_to_longest_side(lung, dose)
    
    if np.max(dose)-np.min(dose) != 0:
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
        plt.imshow(lung[:, :, idx], cmap='gray')
        plt.title(f"Slice {idx}")
        count += 1
        plt.xticks([])
        plt.yticks([])
    for i, idx in enumerate(slice_indices):
        ax = plt.subplot(2, n_slices, count)
        plt.imshow(dose[:, :, idx], cmap='gray')
        count += 1
        plt.xticks([])
        plt.yticks([])

    plt.tight_layout()
    return fig
    
def run_gradcam(model, lung, dose):
    lung_tensor = torch.from_numpy(lung.transpose((2, 0, 1))).float().unsqueeze(0)
    dose_tensor = torch.from_numpy(dose.transpose((2, 0, 1))).float().unsqueeze(0)
    inputs = (lung_tensor * dose_tensor).cuda()
    
    heatmaps = []
    for target_layer in [model.conv4, model.conv3]:
        grad_cam = GradCAM(model, target_layer)
        cam = grad_cam.generate_cam(inputs.unsqueeze(0), target_class=0)
        cam = resize(cam.cpu().numpy()[0], (64, 128, 128))
        hmin, hmax = np.min(cam), np.max(cam)
        heatmap_normalized = (cam - hmin) / (hmax - hmin)
        heatmaps.append(heatmap_normalized[31])
    return tuple(heatmaps)

def colorize_heatmap(heatmap):
    return plt.cm.jet(np.uint8(heatmap * 255))[:, :, :3]

def gradcam_overlay(heatmap, lung, dose, alpha=0.6):
    return heatmap * (1-alpha) + alpha * np.repeat((dose[:, :, 32] * lung[:, :, 32])[:, :, np.newaxis], 3, axis=2)