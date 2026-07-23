import numpy as np
import random
import scipy.io as sio
from sklearn import preprocessing
import matplotlib.pyplot as plt
import os
import logging
import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import Sampler

from torch import Tensor as T
import torch.nn.functional as F
import einops as eo


def load_data(data_path, image_file, label_file):
    image_data = sio.loadmat(os.path.join(data_path, image_file))
    label_data = sio.loadmat(os.path.join(data_path, label_file))

    data_key = image_file.split('/')[-1].split('.')[0]
    label_key = label_file.split('/')[-1].split('.')[0]
    data_all = image_data[data_key]
    GroundTruth = label_data[label_key]

    [nRow, nColumn, nBand] = data_all.shape
    print(data_key, nRow, nColumn, nBand)

    data = data_all.reshape(np.prod(data_all.shape[:2]), np.prod(data_all.shape[2:]))
    data_scaler = preprocessing.scale(data.astype(float))  # (X-X_mean)/X_std
    Data_Band_Scaler = data_scaler.reshape(data_all.shape[0], data_all.shape[1], data_all.shape[2])

    return Data_Band_Scaler, GroundTruth


def load_data_houston(image_file, label_file, label_file1):
    image_data = sio.loadmat(image_file)
    label_data = sio.loadmat(label_file)
    label_data1 = sio.loadmat(label_file1)

    data_key = image_file.split('/')[-1].split('.')[0]
    label_key = label_file.split('/')[-1].split('.')[0]
    label_key1 = label_file1.split('/')[-1].split('.')[0]

    data_all = image_data[data_key]  # dic-> narray , KSC:ndarray(512,217,204)
    GroundTruth_train = label_data[label_key]
    GroundTruth_test = label_data1[label_key1]

    [nRow, nColumn, nBand] = data_all.shape
    print(data_key, nRow, nColumn, nBand)

    data = data_all.reshape(np.prod(data_all.shape[:2]), np.prod(data_all.shape[2:]))  # (111104,204)
    data_scaler = preprocessing.scale(data)  # 标准化 (X-X_mean)/X_std,
    Data_Band_Scaler = data_scaler.reshape(data_all.shape[0], data_all.shape[1], data_all.shape[2])

    return Data_Band_Scaler, GroundTruth_train, GroundTruth_test  # image:(512,217,3),label:(512,217)


def classification_map(map_np: np.array, dpi, savePath):
    height = map_np.shape[0]
    width = map_np.shape[1]
    fig = plt.figure(frameon=False)
    fig.set_size_inches(width * 2.0 / dpi, height * 2.0 / dpi)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    fig.add_axes(ax)
    ax.imshow(map_np)
    fig.savefig(savePath, dpi=dpi)

    return 0


def preprocess(num_ways, num_shots, num_queries, batch_size, device):
    """
    prepare for train and evaluation
    :param num_ways: number of classes for each few-shot task
    :param num_shots: number of samples for each class in few-shot task
    :param num_queries: number of queries for each class in few-shot task
    :param batch_size: how many tasks per batch
    :param device: the gpu device that holds all data
    :return: number of samples in support set
             number of total samples (support and query set)
             mask for edges connect query nodes
             mask for unlabeled data (for semi-supervised setting)
    """
    # set size of support set, query set and total number of data in single task
    num_supports = num_ways * num_shots  # 9 * 1 = 9
    num_samples = num_supports + num_queries * num_ways  # 9 * 1 + 19 * 9 = 180

    # set edge mask (to distinguish support and query edges) 设置边掩码（用于区分支持和查询边）
    support_edge_mask = torch.zeros(batch_size, num_samples, num_samples).to(device)
    support_edge_mask[:, :num_supports, :num_supports] = 1
    query_edge_mask = 1 - support_edge_mask
    evaluation_mask = torch.ones(batch_size, num_samples, num_samples).to(device)  # 作用？mask for unlabeled data (for semi-supervised setting)
    return num_supports, num_samples, query_edge_mask, evaluation_mask


def set_logging_config(logdir, num_seeds):
    myTimeFormat = '%Y-%m-%d_%H-%M-%S'
    nowTime = datetime.datetime.now().strftime(myTimeFormat)

    if not os.path.exists(logdir):
        os.makedirs(logdir)
    logging.basicConfig(format="[%(asctime)s] [%(levelname)s] %(message)s",
                        level=logging.INFO,
                        handlers=[logging.FileHandler(os.path.join(logdir, str(num_seeds) + 'seeds_' + nowTime + '.log')),
                                  logging.StreamHandler(os.sys.stdout)])


device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")


def compute_loss(network_output: torch.Tensor, train_samples_gt_onehot: torch.Tensor, train_label_mask: torch.Tensor):
    real_labels = train_samples_gt_onehot
    we = -torch.mul(real_labels, torch.log(network_output))
    we = torch.mul(we, train_label_mask)
    pool_cross_entropy = torch.sum(we)

    return pool_cross_entropy


def evaluate_performance(network_output, train_samples_gt, train_samples_gt_onehot, zeros):
    with torch.no_grad():
        available_label_idx = (train_samples_gt != 0).float()        # 有效标签的坐标,用于排除背景
        available_label_count = available_label_idx.sum()          # 有效标签的个数
        correct_prediction = torch.where(network_output == torch.argmax(
            train_samples_gt_onehot, 1), available_label_idx, zeros).sum()
        OA = correct_prediction.cpu() / available_label_count
        return OA


# We have used the unfold function from PyTroch instead of manual calculation of the patches in HyperSIGMA.
def get_patch_simple(
    data: T,
    image_height_network: int,
    image_width_network: int,
    overlap_size_height: int,
    overlap_size_width: int
):
    
    num_bands = data.shape[-1]
    
    kernel_size = (image_height_network, image_width_network)
    stride = (image_height_network - overlap_size_height, image_width_network - overlap_size_width)
    
    image_height, image_width = data.shape[:2]
    pad_h = ((stride[0] - (image_height - image_height_network) % stride[0]) % stride[0]) // 2
    pad_w = ((stride[1] - (image_width - image_width_network) % stride[1]) % stride[1]) // 2
    
    # Pad (Left, Right, Top, Bottom)
    data_with_pads = F.pad(data.permute(2, 0, 1), (pad_w, pad_w + 1, pad_h, pad_h + 1), mode='reflect')
    
    # data.shape = [2517, 2335, 20] for the Chikusei dataset
    data_reshaped_with_pads = eo.rearrange(data_with_pads, 'b h w -> 1 b h w').float()  # Its shape becomes: [1, num_bands, H, W]
    
    # Output shape: (num_windows, C * kernel_height * kernel_width, num_patches)
    data_reshaped_with_pads = F.unfold(data_reshaped_with_pads, kernel_size=kernel_size, stride=stride)
    
    data_reshaped_with_pads = eo.rearrange(data_reshaped_with_pads, '1 (h w b) n -> n b h w', h=image_height_network, w=image_width_network, b=num_bands)

    return data_reshaped_with_pads


def get_patch_gt_simple(
    data_gt: T,
    image_height_network: int,
    image_width_network: int,
    overlap_size_height: int,
    overlap_size_width: int
):
    kernel_size = (image_height_network, image_width_network)
    stride = (image_height_network - overlap_size_height, image_width_network - overlap_size_width)
    
    image_height, image_width = data_gt.shape
    pad_h = ((stride[0] - (image_height - image_height_network) % stride[0]) % stride[0]) // 2
    pad_w = ((stride[1] - (image_width - image_width_network) % stride[1]) % stride[1]) // 2
    
    # Pad (Left, Right, Top, Bottom)
    data_gt_reshaped = eo.rearrange(data_gt, 'h w -> 1 1 h w').float()
    data_gt_reshaped_with_pads = F.pad(data_gt_reshaped, (pad_w, pad_w + 1, pad_h, pad_h + 1), mode='constant', value=0)
    
    # Output shape: (num_windows, C * kernel_height * kernel_width, num_patches)
    data_gt_reshaped_with_pads = F.unfold(data_gt_reshaped_with_pads, kernel_size=kernel_size, stride=stride)
    
    # For GT:
    data_gt_reshaped_with_pads = eo.rearrange(data_gt_reshaped_with_pads, '1 (h w) b -> b w h', h=image_height_network, w=image_width_network)
    
    return data_gt_reshaped_with_pads
    
    
def get_patch(image_height_network: int, image_width_network: int, data: T, data_gt: T, Lx, Ly):
    
    # image_size_model_tuple = (image_size_model, image_size_model)
    # height_orgin, width_orgin, bands = data.shape
    N: int = len(Ly) * len(Lx)
    X_data = torch.zeros([N, image_height_network, image_width_network, data.shape[-1]], device=data.device)  # N, H, W, C
    y_data = torch.zeros([N, image_height_network, image_width_network], device=data_gt.device)
    
    i = 0
    
    for j in range(len(Ly)):
        for k in range(len(Lx)):
            row_start = Ly[j]
            col_start = Lx[k]
            row_end, col_end = (row_start + image_height_network, col_start + image_width_network)
            X_data[i] = data[row_start:row_end, col_start:col_end, :]
            y_data[i] = data_gt[row_start:row_end, col_start:col_end]
            i += 1
            
    y_data: T = y_data.type(torch.LongTensor)
    X_data: T = X_data.permute(0, 3, 1, 2).type(torch.FloatTensor)
    
    return X_data, y_data

        
def imshow(data, class_num, name):
    colormap = np.zeros((23, 3))
    colormap[1, :] = [128 / 255, 128 / 255, 128 / 255]
    colormap[2, :] = [0, 255 / 255, 0]
    colormap[3, :] = [0, 255 / 255, 255 / 255]
    colormap[4, :] = [0, 128 / 255, 0]
    colormap[5, :] = [255 / 255, 0, 255 / 255]
    colormap[6, :] = [255 / 255, 255 / 255, 0]
    colormap[7, :] = [0, 0, 128 / 255]
    colormap[8, :] = [255 / 255, 0, 0]
    colormap[9, :] = [128 / 255, 0, 0]
    colormap[10, :] = [0, 0, 255 / 255]
    colormap[11, :] = [237 / 255, 145 / 255, 33 / 255]
    colormap[12, :] = [221 / 255, 160 / 255, 221 / 255]
    colormap[13, :] = [156 / 255, 102 / 255, 31 / 255]
    colormap[14, :] = [125 / 255, 38 / 255, 205 / 255]
    colormap[15, :] = [51 / 255, 161 / 255, 201 / 255]
    colormap[16, :] = [255 / 255, 127 / 255, 80 / 255]
    colormap[17, :] = [128 / 255, 51 / 255, 255 / 255]
    colormap[18, :] = [33 / 255, 128 / 255, 51 / 255]
    colormap[19, :] = [112 / 255, 130 / 255, 255 / 255]
    colormap[20, :] = [237 / 255, 127 / 255, 80 / 255]
    colormap[21, :] = [128 / 255, 237 / 255, 255 / 255]
    colormap[22, :] = [255 / 255, 51 / 255, 128 / 255]
    h, w = data.shape
    truthmap = np.zeros((h, w, 3), dtype=np.float32)
    for k in range(1, class_num + 1):
        for i in range(h):
            for j in range(w):
                if data[i, j] == k:
                    truthmap[i, j, :] = colormap[k, :]
    plt.figure()
    plt.imshow(truthmap)
    plt.axis('off')  # 关闭坐标轴
    plt.savefig(name, dpi=360)
    plt.show()


def imshow_IP(data, class_num, name):
    colormap = np.zeros((23, 3))
    colormap[1, :] = [255 / 255, 0 / 255, 0 / 255]
    colormap[2, :] = [0, 255 / 255, 0]
    colormap[3, :] = [0, 0 / 255, 255 / 255]
    colormap[4, :] = [255 / 255, 255 / 255, 0]
    colormap[5, :] = [0 / 255, 255 / 255, 255 / 255]
    colormap[6, :] = [255 / 255, 0 / 255, 255 / 255]
    colormap[7, :] = [176 / 255, 48 / 255, 96 / 255]
    colormap[8, :] = [46 / 255, 139 / 255, 87 / 255]
    colormap[9, :] = [160 / 255, 32 / 255, 240 / 255]
    colormap[10, :] = [255 / 255, 127 / 255, 80 / 255]
    colormap[11, :] = [127 / 255, 255 / 255, 212 / 255]
    colormap[12, :] = [218 / 255, 112 / 255, 214 / 255]
    colormap[13, :] = [160 / 255, 82 / 255, 45 / 255]
    colormap[14, :] = [127 / 255, 255 / 255, 0 / 255]
    colormap[15, :] = [216 / 255, 191 / 255, 216 / 255]
    colormap[16, :] = [238 / 255, 0 / 255, 0 / 255]
    colormap[17, :] = [128 / 255, 51 / 255, 255 / 255]
    colormap[18, :] = [33 / 255, 128 / 255, 51 / 255]
    colormap[19, :] = [112 / 255, 130 / 255, 255 / 255]
    colormap[20, :] = [237 / 255, 127 / 255, 80 / 255]
    colormap[21, :] = [128 / 255, 237 / 255, 255 / 255]
    colormap[22, :] = [255 / 255, 51 / 255, 128 / 255]
    h, w = data.shape
    truthmap = np.zeros((h, w, 3), dtype=np.float32)
    for k in range(1, class_num + 1):
        for i in range(h):
            for j in range(w):
                if data[i, j] == k:
                    truthmap[i, j, :] = colormap[k, :]
    plt.figure()
    plt.imshow(truthmap)
    plt.axis('off')  # 关闭坐标轴
    plt.savefig(name, dpi=360)
    plt.show()


def imshow_PU(data, class_num, name):
    colormap = np.zeros((23, 3))
    colormap[1, :] = [216 / 255, 191 / 255, 216 / 255]
    colormap[2, :] = [0, 255 / 255, 0]
    colormap[3, :] = [0, 255 / 255, 255 / 255]
    colormap[4, :] = [45 / 255, 138 / 255, 86 / 255]
    colormap[5, :] = [255 / 255, 0 / 255, 255 / 255]
    colormap[6, :] = [255 / 255, 165 / 255, 0 / 255]
    colormap[7, :] = [159 / 255, 31 / 255, 239 / 255]
    colormap[8, :] = [255 / 255, 0 / 255, 0 / 255]
    colormap[9, :] = [255 / 255, 255 / 255, 0 / 255]
    colormap[10, :] = [255 / 255, 127 / 255, 80 / 255]
    colormap[11, :] = [127 / 255, 255 / 255, 212 / 255]
    colormap[12, :] = [218 / 255, 112 / 255, 214 / 255]
    colormap[13, :] = [160 / 255, 82 / 255, 45 / 255]
    colormap[14, :] = [217 / 255, 255 / 255, 0 / 255]
    colormap[15, :] = [216 / 255, 191 / 255, 216 / 255]
    colormap[16, :] = [238 / 255, 0 / 255, 0 / 255]
    colormap[17, :] = [128 / 255, 51 / 255, 255 / 255]
    colormap[18, :] = [33 / 255, 128 / 255, 51 / 255]
    colormap[19, :] = [112 / 255, 130 / 255, 255 / 255]
    colormap[20, :] = [237 / 255, 127 / 255, 80 / 255]
    colormap[21, :] = [128 / 255, 237 / 255, 255 / 255]
    colormap[22, :] = [255 / 255, 51 / 255, 128 / 255]
    h, w = data.shape
    truthmap = np.zeros((h, w, 3), dtype=np.float32)
    for k in range(1, class_num + 1):
        for i in range(h):
            for j in range(w):
                if data[i, j] == k:
                    truthmap[i, j, :] = colormap[k, :]
    plt.figure()
    plt.imshow(truthmap)
    plt.axis('off')  # 关闭坐标轴
    plt.savefig(name, dpi=360)
    plt.show()


def imshow_HC(data, class_num, name):
    colormap = np.zeros((23, 3))
    colormap[1, :] = [255 / 255, 0 / 255, 0 / 255]
    colormap[2, :] = [0, 255 / 255, 0]
    colormap[3, :] = [0, 0 / 255, 255 / 255]
    colormap[4, :] = [255 / 255, 255 / 255, 0]
    colormap[5, :] = [0 / 255, 255 / 255, 255 / 255]
    colormap[6, :] = [255 / 255, 0 / 255, 255 / 255]
    colormap[7, :] = [176 / 255, 48 / 255, 96 / 255]
    colormap[8, :] = [46 / 255, 139 / 255, 87 / 255]
    colormap[9, :] = [160 / 255, 32 / 255, 240 / 255]
    colormap[10, :] = [255 / 255, 127 / 255, 80 / 255]
    colormap[11, :] = [127 / 255, 255 / 255, 212 / 255]
    colormap[12, :] = [218 / 255, 112 / 255, 214 / 255]
    colormap[13, :] = [160 / 255, 82 / 255, 45 / 255]
    colormap[14, :] = [127 / 255, 255 / 255, 0 / 255]
    colormap[15, :] = [216 / 255, 191 / 255, 216 / 255]
    colormap[16, :] = [238 / 255, 0 / 255, 0 / 255]
    colormap[17, :] = [238 / 255, 154 / 255, 0 / 255]
    colormap[18, :] = [85 / 255, 26 / 255, 139 / 255]
    colormap[19, :] = [0 / 255, 139 / 255, 0 / 255]
    colormap[20, :] = [37 / 255, 58 / 255, 150 / 255]
    colormap[21, :] = [47 / 255, 78 / 255, 161 / 255]
    colormap[22, :] = [123 / 255, 18 / 255, 20 / 255]
    h, w = data.shape
    truthmap = np.zeros((h, w, 3), dtype=np.float32)
    for k in range(1, class_num + 1):
        for i in range(h):
            for j in range(w):
                if data[i, j] == k:
                    truthmap[i, j, :] = colormap[k, :]
    plt.figure()
    plt.imshow(truthmap)
    plt.axis('off')  # 关闭坐标轴
    plt.savefig(name, dpi=360)
    plt.show()
    

def Get_train_and_test_data(img_size, img, img_gt):
    H0, W0, C = img.shape
    if H0 < img_size:
        gap = img_size - H0
        mirror_img = img[(H0 - gap):H0, :, :]
        mirror_img_gt = img_gt[(H0 - gap):H0, :]
        img = np.concatenate([img, mirror_img], axis=0)
        img_gt = np.concatenate([img_gt, mirror_img_gt], axis=0)
    if W0 < img_size:
        gap = img_size - W0
        mirror_img = img[:, (W0 - gap):W0, :]
        mirror_img_gt = img_gt[(W0 - gap):W0, :]
        img = np.concatenate([img, mirror_img], axis=1)
        img_gt = np.concatenate([img_gt, mirror_img_gt], axis=1)
    H, W, C = img.shape

    num_H = H // img_size
    num_W = W // img_size
    sub_H = H % img_size
    sub_W = W % img_size
    if sub_H != 0:
        gap = (num_H + 1) * img_size - H
        mirror_img = img[(H - gap):H, :, :]
        mirror_img_gt = img_gt[(H - gap):H, :]
        img = np.concatenate([img, mirror_img], axis=0)
        img_gt = np.concatenate([img_gt, mirror_img_gt], axis=0)

    if sub_W != 0:
        gap = (num_W + 1) * img_size - W
        mirror_img = img[:, (W - gap):W, :]
        mirror_img_gt = img_gt[:, (W - gap):W]
        img = np.concatenate([img, mirror_img], axis=1)
        img_gt = np.concatenate([img_gt, mirror_img_gt], axis=1)
        # gap = img_size - num_W*img_size
        # img = img[:,(W - gap):W,:]
    H, W, C = img.shape
    print('padding img:', img.shape)

    num_H = H // img_size
    num_W = W // img_size
    index = torch.arange(1, H * W + 1)
    index = index.reshape(H, W)
    sub_imgs = []
    sub_indexs = []

    for i in range(num_H):
        for j in range(num_W):
            z = img[i * img_size:(i + 1) * img_size, j *
                    img_size:(j + 1) * img_size, :]
            sub_imgs.append(z)
            w = index[i * img_size:(i + 1) * img_size,
                      j * img_size:(j + 1) * img_size]
            sub_indexs.append(w)
    sub_imgs = np.array(sub_imgs)
    sub_indexs = np.array(sub_indexs)  # [num_H*num_W,img_size,img_size, C ]

    return sub_imgs, sub_indexs, num_H, num_W, img, img_gt


def patch_reshape(pred, num_H, num_W, class_num, img_size):
    pred = torch.reshape(pred, [num_H, num_W, class_num, img_size, img_size])
    pred = torch.permute(pred, [2, 0, 3, 1, 4])  # [2,num_H, img_size,num_W, img_size]]
    pred = torch.reshape(pred, [class_num, num_H * img_size * num_W * img_size])
    pred = torch.permute(pred, [1, 0]) 
    return pred


def image_reshape(y, height, width, height_orgin, width_orgin, class_num):
    y = y.reshape(height, width, class_num)
    y = y[0:height_orgin, 0:width_orgin, :]
    y = y.reshape(height_orgin * width_orgin, class_num)
    return y


class myDataset(torch.utils.data.Dataset):
    def __init__(self, image_hsi: T, ground_truth: T):
        self.len = image_hsi.shape[0]
        self.x_data_hsi = torch.FloatTensor(image_hsi)
        self.y_data = torch.LongTensor(ground_truth)
    
    def __getitem__(self, index):
        # 根据索引返回数据和对应的标签
        return self.x_data_hsi[index], self.y_data[index]

    def __len__(self):
        # 返回文件数据的数目
        return self.len
    
    
def show_number_of_parameters_in_pramas_groups(params_all: list, logger):
    num_parameters = 0
    
    for param_list1 in params_all:
        num_parameters_comp = 0
        
        for p in param_list1['params']:
            if p.requires_grad:
                num_parameters_comp += p.numel()
    
        num_parameters += num_parameters_comp
        logger.info(f"Number of learnable parameters of {param_list1['name']}: {num_parameters_comp}")
    
    logger.info(f'Total number of learnable parameters: {num_parameters}')
    

def synchronize(device):
    if device is not None:
        torch.cuda.synchronize(device)
    
    
# References:
# https://github.com/Naeem-Paeedeh/CVLC
# https://github.com/Qba-heu/FDFSL
# https://github.com/WHU-Sigma/HyperSIGMA
# https://github.com/Li-ZK/CDFS-CASCL-2024
# https://github.com/Naeem-Paeedeh/CPLSR