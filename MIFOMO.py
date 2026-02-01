#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import os
import math
from pathlib import Path

import utils as utils

import json
from model.shared import CoalescentProjection
from model.HyperSIGMA import HyperSIGMA
from torch.distributions import Beta
from torch import Tensor as T
import dataloader as dl
import einops as eo
import new_types as nt
import shared as sh
import logging
import sys
import platform
import torchvision
import GPUtil


class MIFOMO(nn.Module):
    def __init__(self, settings_file: str):
        super().__init__()
        
        self.use_coalescent_projection: bool = True
        self.use_PCA: bool = True
        self.use_mapping: bool = False
        self.use_label_smoothing: bool = True
        self.use_mixup = True
        
        self.settings_file = settings_file
        
        self.checkpoint_file: str = ''
        
        self.gpu_id = 0
        self.device = f'cuda:{self.gpu_id}'
        
        self.alpha = 2.0
        self.beta_distribution = Beta(self.alpha, self.alpha)
        
        self.phase = nt.Phase.Source
        
        self.lr_default = 1e-4
        self.lr_model = 1e-4
        self.lr_mapping = 1e-4
        self.lr_PEFT = 1e-2
        self.lr_temperature_prototypical_classifier = 1e-2
        self.optimizer_name = 'AdamW'
        self.momentum = 0.9
        self.momentum2 = 0.999
        self.dampening = 0.9
        self.weight_decay = 0.05
        
        self.temperature_mixup_scheduler = 0.05
        
        self.seed = 0
        
        self.num_episodes_source = 1000
        self.num_episodes_intermediate_domain = 500
        self.num_episodes_target1 = 1000
        self.num_episodes_target2 = 1000
        self.num_episodes_tests = 20
        
        self.dataset_name_source = 'Chikusei'
        self.dataset_name_target = 'IndianPines'
        self.num_classes_source = 0
        self.num_classes_target = 0
        self.datasets_root_dir = ''
        self.path_source_dataset = ''
        self.path_source_ground_truth = ''
        self.path_target_dataset = ''
        # If path_target_ground_truth is only set (without the path_target_ground_truth_test), we divide it to the train and test (support and query) sets.
        # For the Houston dataset, we use path_target_ground_truth for train and path_target_ground_truth_test for the test set.
        self.path_target_ground_truth = ''
        self.path_target_ground_truth_test = ''
        
        self.excel_file_name = 'Results'    # Without the extension
        
        self.time_str = sh.get_time_str()
        
        self.missed_cache_before = False
        self.disable_caching_mechanism = True
        
        self.dir_log: str = os.path.join('logs', sh.get_time_str(False))
        self.patch_size = 9
        
        self.num_bands_source = 0
        self.num_bands_target = 0
        self.pca_components = 50        # 20
        self.num_bands_after_mapping = 50
        
        self.model: HyperSIGMA = None
        self.mapping_source: nn.Module = None
        self.mapping_target: nn.Module = None
        
        self.num_shots_support = 5  # test_lsample_num_per_class in other codes is the number of support samples for the target domain
        self.num_shots_query = 15
        
        # After pseudo-labeling the target query set, some classes may not have zero samples. However, we have at least the support set samples.
        self.num_shots_support_intermediate_domain = 2
        self.num_shots_query_intermediate_domain = 3
        
        self.num_ways_limit_source = -1  # To limit the memory consumption   # By setting it to less than 2, it would be ignored and the program would use all classes in the dataset as num_ways!
        self.num_ways_source = -1      # It will be set automatically later.
        self.num_ways_target = -1      # It will be set automatically later.
        
        # Mixup
        self.num_mixup_source = -1
        self.num_mixup_target = -1
        self.num_mixup_intermediate = -1
        self.top_k_most_confident = 1e6
        
        self.max_epoch = 500    # Default: 500
        self.batch_size_LGC = 80
        
        self.gt_matrix_target: T = None
        self.gt_matrix_test_target: T = None

        self.coalescent_projection_spat: CoalescentProjection = None
        self.coalescent_projection_spec: CoalescentProjection = None
        
        self.parameter_names_shared = set()                         # Parameter names that can be found both in the checkpoint and the model's state dictionary.
        self.parameter_names_only_available_in_checkpoints = set()  # We can recover them from the checkpoint
        # To optimize. They either do not exist in the checkpoint or we removed them as they must be adapted!
        self.parameter_names_only_available_in_model = set()
        
        self.best_model_dict = {}
        # We use this dictionary to reset the model. We exclude the frozen parameters.
        self.initial_state_dict = {}
        
        self.key_source_dict = ""
        self.key_source_ground_truth_dict = ""
        self.key_target_dict = ""
        self.key_target_ground_truth_dict = ""
        self.key_target_ground_truth_dict_test = ""  # For the test set of the Houston dataset.
        
        self.max_epoch = 31
        self.num_shots_support = 5
        
        self.use_checkpoint: bool = True        # Gradient checkpointing
        
        self.distance_function = nt.DistanceFunction.CosineSimilarity
        
        self.perform_the_intermediate_domain_training = True
        
        # The size of the source and target datasets
        self.height_source_image: int = 0
        self.width_source_image: int = 0
        self.height_target_image: int = 0
        self.width_target_image: int = 0
        
        self.dir_classification_maps = 'classification_map'
        self.dir_saves = 'saves'
        self.dir_cache = 'cache'
        
        # -----------------------------------------------------
        self._load_json_file()
        
        self._prepare_logger()
        
        sh.set_seed(self.seed)
        
        self.samples_source: T = None
        self.labels_source: T = None
        
        self.samples_target: T = None
        self.labels_target: T = None
        self.samples_target_test: T = None
        self.labels_target_test: T = None
        
        os.makedirs('checkpoints', exist_ok=True)
        os.makedirs(self.dir_classification_maps, exist_ok=True)
        os.makedirs(self.dir_cache, exist_ok=True)      # As a temporary cache directory for datasets and intermediate states.
        os.makedirs(os.path.join(self.dir_cache, 'temp'), exist_ok=True)      # As a temporary cache directory.
        os.makedirs(self.dir_saves, exist_ok=True)
        
        assert self.dataset_name_source == 'Chikusei'
        assert self.dataset_name_target in ['PaviaU', 'IndianPines', 'Salinas', 'Houston', 'PaviaC', 'HHK']
        
        self.device = torch.device(self.device if torch.cuda.is_available() else "cpu")
        
        self.path_excel_file = os.path.join(self.dir_log, f'{self.excel_file_name}, {self.time_str}.xlsx')
        
        self.max_epoch = 31
        self.train_num = 5
        
        self._load_datasets()
        self.prototypical_classifier: sh.PrototypicalClassifier = None
        self.prepare_the_model(source_or_target=self.phase == nt.Phase.Source)
        
        if self.checkpoint_file != '':
            self.load(self.checkpoint_file)
            
        self.source_EpisodeGenerator: sh.EpisodeGenerator = None
        self.target_EpisodeGenerator: sh.EpisodeGenerator = None
        
        self._prepare_EpisodeGenerators()
        
        self.print_arguments()
    
    # https://github.com/Naeem-Paeedeh/CONEC-LoRA
    def _load_json_file(self):
        with open(self.settings_file, 'r') as file:
            data: dict = json.load(file)
            
        # Detection and conversion of enum classes
        for k, v in data.items():
            value_final = v
            
            if isinstance(v, str):
                if '.' in v and len(v.split('.')) == 2:    # Enums
                    class_name, enum_value_str = v.split('.')
                    # enum_class = globals().get(class_name)
                    if hasattr(nt, class_name):
                        enum_class = getattr(nt, class_name)
                    
                        if enum_class is not None:
                            value_final = enum_class[enum_value_str]
            
            setattr(self, k, value_final)
                
    def _prepare_EpisodeGenerators(self):

        self.target_EpisodeGenerator = sh.EpisodeGenerator(
            samples=self.samples_target,
            samples_test=self.samples_target_test,
            labels=self.labels_target,
            labels_test=self.labels_target_test,
            num_shots_support=self.num_shots_support,
            num_shots_query=-1,     # We choose all remaining samples following the other methods.
            num_ways_limit=self.num_ways_target,
            device='cpu',
            dataset_name=self.dataset_name_target,
        )
        
        self.source_EpisodeGenerator = sh.EpisodeGenerator(
            samples=self.samples_source,
            labels=self.labels_source,
            num_shots_support=self.num_shots_support,
            num_shots_query=self.num_shots_query,
            num_ways_limit=self.num_ways_source,
            dataset_name=self.dataset_name_source,
            device=self.device,
        )
        
    def generate_an_episode(self, source_or_target: bool):
        
        if source_or_target:
            episode = self.source_EpisodeGenerator.generate()
        else:
            episode = self.target_EpisodeGenerator.generate(keep_permuted_indices=self.phase == nt.Phase.ClassificationMap)
            episode.support_samples = episode.support_samples.to(self.device)
            episode.support_labels = episode.support_labels.to(self.device)
        
        return episode
    
    def suffix_for_dataset_cache_files(self):
        res: str = f',patch_size={self.patch_size}'
        
        if self.use_PCA:
            res += f",pca_components={self.pca_components}"
        else:
            res += "Without_PCA"
            
        res += ".pth"
        
        return res
    
    def suffix_for_files(self, num_episodes: int = -1):
        res: str = f',patch_size={self.patch_size}'
        
        if self.use_PCA:
            res += f",pca_c={self.pca_components}"
        else:
            res += "WO_PCA"
            
        if self.use_mapping:
            res += ",with_mapping"
        else:
            res += ",WO_mapping"
            
        num_episodes_str = ''
            
        if num_episodes == -1:
            if self.phase == nt.Phase.Source:
                num_episodes_str = f'n_ep_src={self.num_episodes_source}'
            else:
                num_episodes_str = f'n_ep_tgt1={self.num_episodes_target1},n_ep_intrm={self.num_episodes_intermediate_domain}'
        else:
            num_episodes_str = f'num_episodes={num_episodes}'
        
        distance_function_str = ''
        if self.distance_function == nt.DistanceFunction.CosineDistance:
            distance_function_str = 'CD'
        elif self.distance_function == nt.DistanceFunction.CosineSimilarity:
            distance_function_str = 'CS'
        elif self.distance_function == nt.DistanceFunction.Euclidean:
            distance_function_str = 'E'
        elif self.distance_function == nt.DistanceFunction.L1:
            distance_function_str = 'L1'
            
        res += f",n_episodes={num_episodes_str},DF={distance_function_str},{self.time_str}.pth"
            
        return res
    
    def _load_datasets(self):
        
        os.makedirs(self.dir_cache, exist_ok=True)
        
        suffix = self.suffix_for_dataset_cache_files()
        
        path_cache_source = os.path.join(self.dir_cache, self.dataset_name_source + suffix)
        path_cache_target = os.path.join(self.dir_cache, self.dataset_name_target + suffix)
    
        if not os.path.exists(path_cache_source):
            image = self._obtain_input_image(source_or_target=True)
            gt_matrix, _ = self._obtain_ground_truth_dataset(source_or_target=True)
            
            self.samples_source, self.labels_source = dl.extract_patches(image, gt_matrix=gt_matrix, patch_size=self.patch_size)
            
            self.gt_matrix_source = gt_matrix
            self.height_source_image = gt_matrix.shape[0]
            self.width_source_image = gt_matrix.shape[1]
            
            torch.save(
                {
                    'samples': self.samples_source,
                    'labels': self.labels_source,
                    'height': gt_matrix.shape[0],
                    'width': gt_matrix.shape[1],
                    'gt_matrix': gt_matrix
                },
                path_cache_source)
        else:
            data_dict = torch.load(path_cache_source)
            self.samples_source = data_dict['samples']
            self.labels_source = data_dict['labels']
            self.gt_matrix_source = data_dict['gt_matrix']
            self.height_source_image = data_dict['height']
            self.width_source_image = data_dict['width']
        
        if not os.path.exists(path_cache_target):
            image = self._obtain_input_image(source_or_target=False)
            gt_matrix, gt_matrix_test = self._obtain_ground_truth_dataset(source_or_target=False)
            
            self.samples_target, self.labels_target = dl.extract_patches(image=image, gt_matrix=gt_matrix, patch_size=self.patch_size)
            
            if self.dataset_name_target == 'Houston':
                self.samples_target_test, self.labels_target_test = dl.extract_patches(image=image, gt_matrix=gt_matrix_test, patch_size=self.patch_size)
                self.gt_matrix_test_target = gt_matrix_test
            
            self.height_target_image = gt_matrix.shape[0]
            self.width_target_image = gt_matrix.shape[1]
            self.gt_matrix_target = gt_matrix
            
            torch.save(
                {
                    'samples': self.samples_target,
                    'samples_test': self.samples_target_test,
                    'labels': self.labels_target,
                    'labels_test': self.labels_target_test,
                    'height': self.height_target_image,
                    'width': self.width_target_image,
                    'gt_matrix': gt_matrix,
                    'gt_matrix_test': gt_matrix_test
                },
                path_cache_target
            )
        else:
            data_dict = torch.load(path_cache_target, map_location='cpu')
            self.samples_target = data_dict['samples']
            self.labels_target = data_dict['labels']
            self.height_target_image = data_dict['height']
            self.width_target_image = data_dict['width']
            self.gt_matrix_target = data_dict['gt_matrix']
            
            if self.dataset_name_target == 'Houston':
                self.samples_target_test = data_dict['samples_test']
                self.labels_target_test = data_dict['labels_test']
                self.gt_matrix_test_target = data_dict['gt_matrix_test']
            
        self.num_classes_source = len(self.labels_source.unique())  # self.labels_source.max().item() + 1
        logging.info(f'num_classes for the {self.dataset_name_source} dataset: {self.num_classes_source}')
        
        self.num_classes_target = len(self.labels_target.unique())  # self.labels_target.max().item() + 1
        logging.info(f'num_classes for the {self.dataset_name_target} dataset: {self.num_classes_target}')
        
        self.num_bands_source = self.samples_source.shape[1]
        self.num_bands_target = self.samples_target.shape[1]
        
        # We set the self.num_ways_source and self.num_ways_target here:
        if self.num_ways_limit_source == -1:
            self.num_ways_limit_source = self.num_classes_source
        
        self.num_ways_source = min(self.num_classes_source, self.num_ways_limit_source)
        self.num_ways_target = self.num_classes_target
        
        # if source_or_target:
        #     self.image_height_source = image_height
        #     self.image_width_source = image_width
        # else:
        #     self.image_height_target = image_height
        #     self.image_width_target = image_width
            
        # assert sum([len(self.label_to_indices_source[lbl]) for lbl in range(self.num_classes_source)]) == len(self.labels_source)
        # assert sum([len(self.label_to_indices_target[lbl]) for lbl in range(self.num_classes_target)]) == len(self.labels_target)
            
    def _obtain_input_image(self, source_or_target: bool):
        if source_or_target:
            dataset_name = self.dataset_name_source
            path_dataset = os.path.join(self.datasets_root_dir, self.path_source_dataset)
            key_in_data_dict = self.key_source_dict
        else:
            dataset_name = self.dataset_name_target
            path_dataset = os.path.join(self.datasets_root_dir, self.path_target_dataset)
            key_in_data_dict = self.key_target_dict
        
        device = self.device

        # data.shape: [145, 145, 200] for Indian pines
        # data_gt.shape: [145, 145] for Indian pines
        image = dl.load_input_image(path_dataset=path_dataset, key_dataset_dict=key_in_data_dict)
        
        print(f'Image shape for the {dataset_name} dataset: {image.shape}')
        
        assert len(image.shape) == 3
        
        image_height, image_width, num_bands = image.shape
        
        if self.use_PCA and self.pca_components < num_bands:
            image = dl.apply_PCA(image, num_components=self.pca_components)
        
        image = torch.from_numpy(image)
        
        if source_or_target:
            image = image.to(device)
        
        image = eo.rearrange(image, 'h w c -> c h w')
        
        print(f'data.shape after PCA: {image.shape}')   # data.shape: [20, image_heght, image_width]
        
        image = dl.preprocess_data(image)
        
        return image    # data.shape: [20, image_heght, image_width]
        
    def _obtain_ground_truth_dataset(self, source_or_target: bool):
        if source_or_target:
            dataset_name = self.dataset_name_source
            path_ground_truth = os.path.join(self.datasets_root_dir, self.path_source_ground_truth)
            key_in_gt_dict = self.key_source_ground_truth_dict
        else:
            dataset_name = self.dataset_name_target
            path_ground_truth = os.path.join(self.datasets_root_dir, self.path_target_ground_truth)
            key_in_gt_dict = self.key_target_ground_truth_dict
            
            if dataset_name == 'Houston':
                path_ground_truth_test = os.path.join(self.datasets_root_dir, self.path_target_ground_truth_test)
                key_in_gt_dict_test = self.key_target_ground_truth_dict_test
        
        # data.shape: [145, 145, 200] for Indian pines
        # data_gt.shape: [145, 145] for Indian pines
        gt_matrix = dl.load_ground_truth_matrix(dataset_name=dataset_name, path_dataset_ground_truth=path_ground_truth, key_in_gt_dict=key_in_gt_dict)
        
        gt_matrix = torch.from_numpy(gt_matrix).long()
        
        gt_matrix_test: T = None
        if dataset_name == 'Houston':
            gt_matrix_test = dl.load_ground_truth_matrix(dataset_name=dataset_name, path_dataset_ground_truth=path_ground_truth_test, key_in_gt_dict=key_in_gt_dict_test)
            
            gt_matrix_test = torch.from_numpy(gt_matrix_test).long()
            print(f'Ground truth matrix shape for the {dataset_name} dataset: {gt_matrix.shape}')
        
        # image_height, image_width = gt_matrix.shape
        
        # if source_or_target:
        #     assert self.image_height_source == image_height
        #     assert self.image_width_source == image_width
        #     # self.gt_matrix_source = gt_matrix
        #     self.num_classes_source = num_classes
        # else:
        #     assert self.image_height_target == image_height
        #     assert self.image_width_target == image_width
        #     # self.gt_matrix_target = gt_matrix
        #     self.num_classes_target = num_classes
        
        return gt_matrix, gt_matrix_test
        
    def reinitialize_the_coalescent_projecitons(self):
        if self.use_coalescent_projection:
            self.coalescent_projection_spat = CoalescentProjection(dim_embed=self.model.spat_encoder.dim_embed, num_heads=self.model.spat_encoder.num_heads, num_layers=self.model.spat_encoder.num_layers, device=self.device)
            self.coalescent_projection_spec = CoalescentProjection(dim_embed=self.model.spec_encoder.dim_embed, num_heads=self.model.spat_encoder.num_heads, num_layers=self.model.spat_encoder.num_layers, device=self.device)
    
    # https://github.com/WHU-Sigma/HyperSIGMA
    def prepare_the_model(self, source_or_target: bool):
        
        if source_or_target:
            num_classes = self.num_classes_source
        else:
            num_classes = self.num_classes_target
    
        assert num_classes > 0
        
        # spat_net = torch.load((r"spat-base.pth"), map_location=torch.device('cpu'))
        spat_net_dict = torch.load((r"pre-trained/spat-vit-base-ultra-checkpoint-1599.pth"), map_location=torch.device('cpu'), weights_only=False)
        
        # model_params['spat_encoder.pos_embed'].shape = [1, 256, 768]
        # spat_net_dict['model']['pos_embed'].shape    = [1,  64, 768]
        
        for k in list(spat_net_dict['model'].keys()):
            if 'patch_embed.proj' in k:
                del spat_net_dict['model'][k]
        # The following parts may not exist!
        for k in list(spat_net_dict['model'].keys()):
            if 'spat_map' in k:
                del spat_net_dict['model'][k]
        for k in list(spat_net_dict['model'].keys()):
            if 'spat_output_maps' in k:
                del spat_net_dict['model'][k]
        for k in list(spat_net_dict['model'].keys()):
            if 'pos_embed' in k:
                del spat_net_dict['model'][k]
                
                # # target_num_tokens = (self.image_height_network // self.patch_size_ViT) * (self.image_width_network // self.patch_size_ViT)  # 256 patches (16x16)
                # target_num_tokens = self.patch_size[0] * self.patch_size[1]
                
                # # Bicubic interpolation
                # # Get the pretrained embedding
                # pos_embed = spat_net_dict['model'][k]  # Shape: [1, 64, 768]
                
                # if pos_embed.shape[1] != target_num_tokens:
                #     # print(f"Interpolating pos_embed from {posemb.shape[1]} to {target_num_patches}...")
                    
                #     # 3. Calculate grid sizes
                #     orig_size = int(math.sqrt(pos_embed.shape[1]))
                #     new_size = int(math.sqrt(target_num_tokens))
                    
                #     # Reshape to [1, C, H, W] for grid_sample/interpolate
                #     # Current: [1, N, C] -> Reshape [1, H, W, C] -> Permute [1, C, H, W]
                #     posemb_grid = pos_embed.reshape(1, orig_size, orig_size, -1).permute(0, 3, 1, 2)
                    
                #     posemb_grid = F.interpolate(
                #         posemb_grid,
                #         size=(new_size, new_size),
                #         mode='bicubic',
                #         align_corners=False
                #     )
                    
                #     # Flatten back to [1, N, C]
                #     # [1, C, H, W] -> Permute [1, H, W, C] -> Flatten [1, N, C]
                #     new_posemb = posemb_grid.permute(0, 2, 3, 1).flatten(1, 2)
                    
                #     spat_net_dict['model'][k] = new_posemb
        
        spat_weights = {}
        prefix = 'spat_encoder.'
        
        for key, value in spat_net_dict['model'].items():
            new_key = prefix + key
            spat_weights[new_key] = value
            
        # per_net = torch.load((r"spec-base.pth"), map_location=torch.device('cpu'))
        spec_net_dict = torch.load((r"pre-trained/spec-vit-base-ultra-checkpoint-1599.pth"), map_location=torch.device('cpu'), weights_only=False)

        for k in list(spec_net_dict['model'].keys()):
            if 'patch_embed.proj' in k:
                del spec_net_dict['model'][k]
            if 'spat_map' in k:
                del spec_net_dict['model'][k]
            if 'fpn1.0.weight' in k:
                del spec_net_dict['model'][k]
        
        spec_weights = {}
        prefix = 'spec_encoder.'
        
        for key, value in spec_net_dict['model'].items():
            new_key = prefix + key
            spec_weights[new_key] = value
        
        for k in list(spec_weights.keys()):
            if 'spec_encoder.patch_embed' in k:
                del spec_weights[k]
        
        merged_params = {**spat_weights, **spec_weights}
        
        self.parameter_names_shared = set()                         # To load
        self.parameter_names_only_available_in_checkpoints = set()  # To keep frozen
        self.parameter_names_only_available_in_model = set()        # To optimize
        
        model = HyperSIGMA(
            patch_size=self.patch_size,
            num_bands=self.pca_components,
            num_classes=num_classes,
            model_size='base',  # The optional values are 'base','large' and 'huge',
            use_classifier_head=False,
            use_checkpoint=self.use_checkpoint
        )
        
        if self.use_mapping:
            self.mapping_source = nn.Conv2d(self.num_bands_source, self.num_bands_after_mapping, kernel_size=1, device=self.device)
            self.mapping_target = nn.Conv2d(self.num_bands_target, self.num_bands_after_mapping, kernel_size=1, device=self.device)
        
        model_params = model.state_dict()
        keys_union = set(merged_params.keys()) | set(model_params.keys())
        temp = list(keys_union)
        
        for key in temp:
            if '.decoder_' in key:
                keys_union.remove(key)
        
        for key in keys_union:
            flag_exists_in_model = False
            flag_exists_in_checkpoint = False
            
            if key in model_params.keys():
                flag_exists_in_model = True
            
            if key in merged_params.keys():
                flag_exists_in_checkpoint = True
            
            if (flag_exists_in_model and flag_exists_in_checkpoint):        # or (self.phase == nt.Phase.Intermediate and '.pos_embed' in key)
                self.parameter_names_shared.add(key)
            elif flag_exists_in_model:
                self.parameter_names_only_available_in_model.add(key)
            elif flag_exists_in_checkpoint:
                self.parameter_names_only_available_in_checkpoints.add(key)
            else:   # Redundant!
                raise NotImplementedError
        
        # params_available = {k: v for k, v in merged_params.items() if k in model_params.keys()}
        params_available = {k: v for k, v in merged_params.items() if k in self.parameter_names_shared}
        model_params.update(params_available)
        model.load_state_dict(model_params)
        
        self.model = model
        
        self.reinitialize_the_coalescent_projecitons()
        
        self._freeze_or_unfreeze_the_required_components()
        
        self.prototypical_classifier: sh.PrototypicalClassifier = sh.PrototypicalClassifier(distance_function=self.distance_function)
        
        self.initial_state_dict = self.obtain_parameters_dictionary_to_save()
        
        self.model = self.model.to(self.device)
        
    def reset_the_model(self):
        self._load_dict(self.initial_state_dict)
        self._freeze_or_unfreeze_the_required_components()
        self.missed_cache_before = False
        
    def _load_dict(self, state_dict: dict):
        if self.use_mapping and 'mapping_source' in state_dict.keys() or 'mapping_target' in state_dict.keys():
            self.mapping_source.load_state_dict(state_dict['mapping_source'])
            
            # We do not want to train it from scratch:
            if self.use_PCA:
                self.mapping_target.load_state_dict(state_dict['mapping_source'])
            else:
                raise NotImplementedError
            
        if self.use_coalescent_projection:
            assert 'coalescent_projection_spat' in state_dict.keys() and 'coalescent_projection_spec' in state_dict.keys()
            self.coalescent_projection_spat.load_state_dict(state_dict['coalescent_projection_spat'])
            self.coalescent_projection_spec.load_state_dict(state_dict['coalescent_projection_spec'])
            
        self.prototypical_classifier.load_state_dict(state_dict['prototypical_classifier'])
            
        self.model.load_state_dict(state_dict, strict=False)
    
    def load(self, file_path: str):
        parameters_dict: dict = torch.load(file_path, weights_only=False)
        
        self._load_dict(parameters_dict)
        
        self._freeze_or_unfreeze_the_required_components()
        
        logging.info(f'"{file_path}" is loaded!')
        
    def save(self, file_name: str, num_episodes: int = -1, cache: bool = False, **kwargs):
        model_dict = self.obtain_parameters_dictionary_to_save()
        
        model_dict['extra_details'] = kwargs
        
        if cache:
            if file_name is None or self.disable_caching_mechanism:
                return
            path = os.path.join(self.dir_cache, 'temp', file_name)
        else:
            file_name += self.suffix_for_files(num_episodes=num_episodes)
            path = os.path.join(self.dir_saves, file_name)
        
        torch.save(model_dict, path)
        
        logging.info(f'The model is saved in the "{path}"')
        
    def try_cache_first(self, file_path: str = None):
        if file_path is None or file_path == '' or self.disable_caching_mechanism:
            return False
        file_path = os.path.join(self.dir_cache, 'temp', file_path)
        
        if not self.missed_cache_before and os.path.exists(file_path):
            self.load(file_path)
            return True
        
        self.missed_cache_before = True     # After this, we must ignore the cache files. Therfore, the program overwrite the next cache files!
        return False
        
    def obtain_parameters_dictionary_to_save(self, ignore_frozen_parameters: bool = True) -> dict:
        # We exclude the blocks as they are frozen
        parameters_dict_to_save = self.model.state_dict()
        keys = list(parameters_dict_to_save.keys())
        
        # We do not save the blocks because they are frozen
        if ignore_frozen_parameters:
            for key in keys:
                # if 'blocks.' in key:
                if not self.model.get_parameter(key).requires_grad:
                    parameters_dict_to_save.pop(key)
                    
        if self.use_mapping:
            parameters_dict_to_save['mapping_source'] = self.mapping_source.state_dict()
            parameters_dict_to_save['mapping_target'] = self.mapping_target.state_dict()
            
        if self.use_coalescent_projection:
            parameters_dict_to_save['coalescent_projection_spat'] = self.coalescent_projection_spat.state_dict()
            parameters_dict_to_save['coalescent_projection_spec'] = self.coalescent_projection_spec.state_dict()
            
        parameters_dict_to_save['prototypical_classifier'] = self.prototypical_classifier.state_dict()
        
        return parameters_dict_to_save
    
    def forward(self, x: T, source_or_target: bool):        # source_or_target is just for mapping.
        # x.shape = [batch_size, num_bands, patch_heigh, patch_width]
        x = x.to(self.device)
        
        if self.use_mapping:
            if source_or_target:    # Source
                x = self.mapping_source.forward(x)
            else:                   # Target
                x = self.mapping_target.forward(x)
            
        x = self.model.forward(x, coalescent_projection_spat=self.coalescent_projection_spat, coalescent_projection_spec=self.coalescent_projection_spec)
        
        return x
        
    # https://github.com/Naeem-Paeedeh/CPLSR
    def _freeze_or_unfreeze_the_required_components(self, learnable_HyperSIGMA_parameters: bool = True):
            
        def it_is_not_learnable(model):
            return model is None or type(model) in [int, float, list, dict]
        
        def freeze_or_unfreeze_parameters(model, requires_grad: bool):
            if it_is_not_learnable(model):
                return
            if isinstance(model, nn.Parameter):
                model.requires_grad = requires_grad
            else:
                for param in model.parameters():
                    if hasattr(param, 'requires_grad'):
                        param.requires_grad = requires_grad
                
        def freeze_or_unfreeze_with_exceptions(module: nn.Module, requires_grad: bool, excp_substring_list: list):
            if it_is_not_learnable(module):
                return
            for name, param in module.named_parameters():
                for substr in excp_substring_list:
                    if substr not in name and hasattr(param, 'requires_grad'):
                        param.requires_grad = requires_grad
        
        def freeze_or_unfreeze_with_substring(model: nn.Module, requires_grad: bool, substring_list: list):
            if it_is_not_learnable(model):
                return
            for name, param in model.named_parameters():
                for substr in substring_list:
                    if substr in name and hasattr(param, 'requires_grad'):
                        param.requires_grad = requires_grad
        
        def freeze_or_unfreeze_with_type(model: nn.Module, requires_grad: bool, types_list: list):
            if it_is_not_learnable(model):
                return
            for module in model.modules():
                if type(module) in types_list:
                    for param in module.parameters():
                        if hasattr(param, 'requires_grad'):
                            param.requires_grad = requires_grad
                            
        freeze_or_unfreeze_parameters(model=self.model, requires_grad=False)
        
        # We unfreeze the randomly initialized parameters.
        if self.phase == nt.Phase.Source:
            freeze_or_unfreeze_with_substring(model=self.model, substring_list=self.parameter_names_only_available_in_model, requires_grad=learnable_HyperSIGMA_parameters)  # '.pos_embed'
        elif self.phase == nt.Phase.Intermediate:
            names_list = self.parameter_names_only_available_in_model.copy()
            
            for name in self.parameter_names_only_available_in_model:
                # ['merger.bias', 'spec_encoder.l1.weight', 'merger.weight']
                #  or 'mlp_spec' in name # '.patch_embed' in name  # 'DR' in name
                if '.pos_embed' in name or '.spat_map' in name or 'merger.' in name or 'spec_encoder.l1.' in name:
                    names_list.remove(name)
            
            freeze_or_unfreeze_with_substring(
                model=self.model,
                substring_list=names_list,
                requires_grad=learnable_HyperSIGMA_parameters
            )  # '.pos_embed'
        
        if self.use_coalescent_projection:
            freeze_or_unfreeze_parameters(model=self.coalescent_projection_spat, requires_grad=True)
            freeze_or_unfreeze_parameters(model=self.coalescent_projection_spec, requires_grad=True)
            
        if self.use_mapping:
            freeze_or_unfreeze_parameters(model=self.mapping_source, requires_grad=True)
            freeze_or_unfreeze_parameters(model=self.mapping_target, requires_grad=True)
        
        freeze_or_unfreeze_parameters(model=self.prototypical_classifier, requires_grad=self.lr_temperature_prototypical_classifier > 0.0)
    
    def obtain_the_optimizer(
        self,
        component_additional: nn.Module | nn.Parameter = None,
        lr_component: float = 0.0
    ):
        
        def get_params_groups(model: nn.Module, name_model: str, lr=-1, force_considering_as_non_reqularized=False):
            if model is None or lr == 0:
                return []
            regularized = []
            not_regularized = []
            
            if isinstance(model, nn.Parameter):
                not_regularized.append(model)
            else:
                for name, param in model.named_parameters():
                    if not param.requires_grad:
                        continue
                    # we do not regularize biases nor Norm parameters
                    
                    flag_not_regularized = False
                    
                    if name.endswith(".bias") or 'cls_token' in name or param.dim() == 1 or param.numel() == 1 or force_considering_as_non_reqularized:      # param.numel() == 1 or # 'pos_embed' in name or 'cls_token' in name
                        flag_not_regularized = True
                    
                    if flag_not_regularized:
                        not_regularized.append(param)
                    else:
                        regularized.append(param)
            
            regularized_dict = {'params': regularized, 'name': name_model + ' regularized'}
            not_regularized_dict = {'params': not_regularized, 'weight_decay': 0., 'name': name_model + ' not_regularized'}
            
            if lr != -1:
                regularized_dict['lr'] = lr
                not_regularized_dict['lr'] = lr
            else:
                raise NotImplementedError
                
            result = []
            if len(regularized_dict['params']) > 0:
                result.append(regularized_dict)
            
            if len(not_regularized_dict['params']) > 0:
                result.append(not_regularized_dict)
            
            return result
        
        # self._freeze_or_unfreeze_the_required_components()
        
        params_all = []
        
        params_all += get_params_groups(self.model, lr=self.lr_model, name_model='ViTs', force_considering_as_non_reqularized=False)
        
        params_all += get_params_groups(self.coalescent_projection_spat, lr=self.lr_PEFT, name_model='coalescent_projection_spat', force_considering_as_non_reqularized=True)
        
        params_all += get_params_groups(self.coalescent_projection_spec, lr=self.lr_PEFT, name_model='coalescent_projection_spec', force_considering_as_non_reqularized=True)
        
        params_all += get_params_groups(self.prototypical_classifier, lr=self.lr_temperature_prototypical_classifier, name_model='prototypical_classifier.temperature')
        
        if self.use_mapping:
            params_all += get_params_groups(self.mapping_source, lr=self.lr_mapping, name_model="mapping_source")
            params_all += get_params_groups(self.mapping_target, lr=self.lr_mapping, name_model="mapping_target")
            
        if component_additional is not None:
            params_all += get_params_groups(component_additional, lr=lr_component, name_model="Additional component")
            raise NotImplementedError
            
        optimizer = None
            
        if len(params_all) > 0:
        
            if self.optimizer_name == 'SGD':
                optimizer = torch.optim.SGD(
                    lr=self.lr_default,
                    params=params_all,
                    momentum=self.momentum,
                    dampening=self.dampening,
                    weight_decay=self.weight_decay
                )
            elif self.optimizer_name == 'Adam':
                optimizer = torch.optim.Adam(
                    lr=self.lr_default,
                    params=params_all,
                    betas=(self.momentum, self.momentum2),
                    weight_decay=self.weight_decay,
                    eps=1e-4,
                )
            elif self.optimizer_name == 'AdamW':
                optimizer = torch.optim.AdamW(
                    lr=self.lr_default,
                    params=params_all,
                    betas=(self.momentum, self.momentum2),
                    weight_decay=self.weight_decay,
                    eps=1e-4,
                )
            else:
                raise NotImplementedError
        
        return optimizer
    
    def train(self, mode: bool = True):
        super().train(mode)
        
        self.model.train(mode)
        
        return self
    
    def _prepare_logger(self):
        os.makedirs(self.dir_log, exist_ok=True)
        
        if self.phase == nt.Phase.Source:
            phase_str = 'Phase1'
        elif self.phase == nt.Phase.Intermediate:
            phase_str = 'Phase2'
        elif self.phase == nt.Phase.ClassificationMap:
            phase_str = 'Classification_Map'
        elif self.phase == nt.Phase.t_SNE:
            phase_str = 't_SNE'
        else:
            raise NotImplementedError
            
        log_file_name = os.path.join(self.dir_log, f"{phase_str},DS_t={self.dataset_name_target},{self.suffix_for_files()},Seed={self.seed},Time={self.time_str}")
        
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(filename)s:%(lineno)d] => %(message)s",
            handlers=[
                logging.FileHandler(filename=log_file_name + ".log"),
                logging.StreamHandler(sys.stdout),
            ],
        )
        
    def print_arguments(self):
        
        printable_types_of_attributes = {int, float, str, list, bool, tuple, dict, torch.device, Path}
        
        # GPU
        logging.info(sh.highlighted_message('GPU'))
        gpus = GPUtil.getGPUs()
        current_gpu = gpus[self.gpu_id]
        # logging.info(f"ID: {current_gpu.id}")
        logging.info(f'Python version: {platform.python_version()}')
        logging.info(f'PyTorch version: {torch.__version__},TorchVision version: {torchvision.__version__}')
        logging.info(f"GPU Name: {current_gpu.name}")
        logging.info(f"Driver: {current_gpu.driver}")
        logging.info('-' * 80)
        
        attributes = self.__dict__
        ignore_set = {'args', 'log_file', 'split_to_tar_file'}

        logging.info(sh.highlighted_message("The given arguments"))
        
        wait_list = ['device']
        
        def print_name_and_values(name, value):
            if name == 'device':
                logging.info(f"{name} = \"%s\"", str(value))
            elif type_attr is not str:
                logging.info("%s = %s" % (name, str(value)))
            else:
                logging.info('%s = "%s"' % (name, value))

        for name in attributes.keys():
            value = getattr(self, name)
            type_attr = type(value)
            if name.startswith("_") or name in ignore_set or name in wait_list or type_attr not in printable_types_of_attributes or hasattr(nt, name):        # or value is None
                continue
            
            print_name_and_values(name, value)

        # We show these settings at last.
        for name in wait_list:
            value = getattr(self, name)
            print_name_and_values(name, value)
            
        logging.info('-' * 80)
        

class MixupScheduler:
    def __init__(
        self,
        total_iterations: int,
        temperature: float,
        perturbation_range: float = 0.2,  # Sigma in the paper
        epsilon: float = 1e-8,
    ):
        self.lmbda = 0.0     # It should be close to the target domain at the beginning.
        self.total_iterations = total_iterations
        self.temperature = temperature
        self.perturbation_range = perturbation_range
        self.epsilon = epsilon
    
    @torch.no_grad()
    def update(self, step: int, embeddings_source: T, embeddings_target: T, embeddings_intermediate: T, normalize: bool = True):
        
        if normalize:
            embeddings_source = F.normalize(embeddings_source, dim=1)
            embeddings_target = F.normalize(embeddings_target, dim=1)
            embeddings_intermediate = F.normalize(embeddings_intermediate, dim=1)
            
        dist_source_intermediate = sh.W_distance(embeddings_source, embeddings_intermediate)
        dist_target_intermediate = sh.W_distance(embeddings_target, embeddings_intermediate)
        
        denom = (dist_source_intermediate + dist_target_intermediate) * self.temperature + self.epsilon
        
        q = math.exp(-1.0 * dist_source_intermediate / denom)
        
        assert step <= self.total_iterations
        
        # q = math.exp(-1 * wasserstein_distance(features_source, features_mixup) / (wasserstein_distance(features_source, features_mixup) + wasserstein_distance(features_target, features_mixup) * temperature))      # Eq. 18
    
        # lbd_mix = (i * (1 - q) / num_iter) + q * lbd_mix      # Eq. 19
        # lbd_mix = torch.clamp(torch.as_tensor(random.uniform(lbd_mix - 0.2, lbd_mix + 0.2)), min=0.0, max=1.0)    # Eq. 20
        
        self.lmbda = ((step / self.total_iterations) * (1.0 - q)) + q * self.lmbda
        
    def sample_lambda(self, num_samples: int) -> float:
        rnd = np.random.uniform(low=self.lmbda - self.perturbation_range, high=self.lmbda + self.perturbation_range, size=num_samples)
        rnd = torch.as_tensor(rnd).float()
        lmbda_tilde = torch.clamp(rnd, min=0.0, max=1.0)
        return lmbda_tilde
    

# We have used some codes from the implementation of the following references and tried to conform to the conditions in their experiments:

# https://github.com/WHU-Sigma/HyperSIGMA
# @article{wang2025hypersigma,
#   title={Hypersigma: Hyperspectral intelligence comprehension foundation model},
#   author={Wang, Di and Hu, Meiqi and Jin, Yao and Miao, Yuchun and Yang, Jiaqi and Xu, Yichu and Qin, Xiaolei and Ma, Jiaqi and Sun, Lingyu and Li, Chenxing and others},
#   journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
#   year={2025},
#   publisher={IEEE}
# }

# https://github.com/Qba-heu/FDFSL
# @article{qin2024cross,
#   title={Cross-domain few-shot learning based on feature disentanglement for hyperspectral image classification},
#   author={Qin, Boao and Feng, Shou and Zhao, Chunhui and Li, Wei and Tao, Ran and Xiang, Wei},
#   journal={IEEE Transactions on Geoscience and Remote Sensing},
#   volume={62},
#   pages={1--15},
#   year={2024},
#   publisher={IEEE}
# }

# https://github.com/Naeem-Paeedeh/CPLSR
# @article{paeedeh2025cross,
#   title={Cross-Domain Few-Shot Learning with Coalescent Projections and Latent Space Reservation},
#   author={Paeedeh, Naeem and Pratama, Mahardhika and Mayer, Wolfgang and Cao, Jimmy and Kowlczyk, Ryszard},
#   journal={arXiv preprint arXiv:2507.15243},
#   year={2025}
# }

# https://github.com/Naeem-Paeedeh/CONEC-LoRA
