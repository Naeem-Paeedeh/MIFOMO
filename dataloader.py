# https://github.com/Qba-heu/FDFSL
# https://github.com/WHU-Sigma/HyperSIGMA/blob/main/ImageClassification/demo%20seg%20hypersigma.ipynb

import torch
import numpy as np
import scipy.io as sio
import einops as eo
import torch.nn.functional as F
from torch import Tensor as T
from sklearn.decomposition import PCA
import hdf5storage


def apply_PCA(X, num_components=75):
    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=num_components, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0], X.shape[1], num_components))
    return newX
   
    
def load_matlab_file(path_dataset: str, key: str):
    try:
        data = sio.loadmat(path_dataset)
        if key not in data.keys():
            print(data.keys())
            raise f"Key {key} does not exist in the {path_dataset} file!"
        data_final = data[key]
    except Exception as e:
        data = hdf5storage.loadmat(path_dataset)
        data_final = data[key][:]
    
    return data_final


def load_input_image(path_dataset: str, key_dataset_dict: str):
    data = load_matlab_file(path_dataset, key_dataset_dict)
    data = data.astype(np.float32)
    
    return data


def load_ground_truth_matrix(dataset_name: str, path_dataset_ground_truth: str, key_in_gt_dict: str):
    data_gt = load_matlab_file(path_dataset_ground_truth, key_in_gt_dict)
    
    if dataset_name == 'Chikusei' and key_in_gt_dict == 'GT':
        data_gt = data_gt[0][0][0]
        
    # data_gt = data_gt.astype(np.float32)
    data_gt = data_gt.astype(np.int32)
    
    return data_gt


def standardize(x: T, dim=0, eps=1e-8):
    # dim=0 => per-feature scaling for shape [N, D]
    mean = x.mean(dim=dim, keepdim=True)
    std  = x.std(dim=dim, keepdim=True, correction=0)
    return (x - mean) / (std + eps)


# We follow the https://github.com/Qba-heu/FDFSL
def preprocess_data(image: T):
    shape = image.shape
    num_bands = shape[0]
    image_reshaped = image.reshape(num_bands, -1)
    data_standardized = standardize(image_reshaped, dim=-1)
    data_final = data_standardized.reshape(*shape)
    return data_final


# We follow the https://github.com/Qba-heu/FDFSL
def extract_patches(image: T, gt_matrix: T, patch_size: int = 9, pad_mode="reflect", pad_filling_value=0):
    # image.shape: (num_bands, H, W)
    # gt_matrix.shape: (H, W)
    
    assert patch_size > 0 and patch_size % 2 == 1, "Error: Patch size must be an odd number!"

    centers = torch.nonzero(gt_matrix > 0, as_tuple=False)
    
    labels = gt_matrix[centers[:, 0], centers[:, 1]] - 1    # We subtract 1 to start the labels from zero.
    
    # We remap the labels to have consecutive labels that start from zero.
    labels_unique_list = labels.unique().tolist()
    
    labels_remapped = torch.zeros_like(labels)
    
    for label_new, label_old in enumerate(labels_unique_list):
        labels_remapped[labels == label_old] = label_new
    
    r = patch_size // 2     # Radius of patches
    
    # Pad Image:
    # F.pad arguments are reversed: (last_dim_left, last_dim_right, 2nd_last_left, ...)
    # Tuple format: (pad_C_left, pad_C_right, pad_W_left, pad_W_right, pad_H_left, pad_H_right)
    padded_image = F.pad(image, (r, r, r, r), mode=pad_mode, value=pad_filling_value)     # (H + 2 * r, W + 2 * r, num_bands)
    
    # Adjusting the centers and creating the grid:
    centers_shifted = centers + r
    offset_range = torch.arange(-r, r + 1)
    r_offsets, c_offsets = torch.meshgrid(offset_range, offset_range, indexing='ij')
    
    # Computing the absolute indices
    # Output grid shape: (N, patch_size, patch_size)
    grid_rows = centers_shifted[:, 0].view(-1, 1, 1) + r_offsets.view(1, patch_size, patch_size)
    grid_cols = centers_shifted[:, 1].view(-1, 1, 1) + c_offsets.view(1, patch_size, patch_size)
    
    # Indexing with the grids:
    # padded_image = eo.rearrange(padded_image, 'c h w -> h w c')
    patches = padded_image[:, grid_rows, grid_cols]
    patches = eo.rearrange(patches, 'c s h w -> s c h w')
    
    return patches, labels_remapped      # Patches of shape (num_total_samples, num_bands, patch_size, patch_size)
