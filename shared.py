# We borrowed the codes from the following repository (or repositories):
# https://github.com/Naeem-Paeedeh/CPLSR
# @article{paeedeh2025cross,
#   title={Cross-Domain Few-Shot Learning with Coalescent Projections and Latent Space Reservation},
#   author={Paeedeh, Naeem and Pratama, Mahardhika and Mayer, Wolfgang and Cao, Jimmy and Kowlczyk, Ryszard},
#   journal={arXiv preprint arXiv:2507.15243},
#   year={2025}
# }

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import new_types as nt
from torch import Tensor as T
import einops as eo
import random
import os
import collections
import shutil
import gc
import time
import shared as sh
import logging
# from scipy.stats import wasserstein_distance
import ot


# https://github.com/Naeem-Paeedeh/CPLSR
def to_device(input: list | tuple | str | int | float, device):
    if torch.is_tensor(input):
        return input.to(device=device, non_blocking=True)
    elif isinstance(input, str) or isinstance(input, int) or isinstance(input, float) or input is None:
        return input
    elif isinstance(input, collections.abc.Mapping):
        return {k: to_device(sample, device=device) for k, sample in input.items()}
    elif isinstance(input, collections.abc.Sequence):
        return [to_device(sample, device=device) for sample in input]
    else:
        raise TypeError("Input must contain tensor, dict or list, found {type(input)}")


def make_pair(x):
    # if not isinstance(x, tuple) and not isinstance(x, list):
    if type(x) in [int, float]:
        x = (x, x)
    elif type(x) in [tuple, list]:
        if len(x) != 2:
            raise NotImplementedError
    else:
        raise NotImplementedError
    return x


def get_time_str(add_time: bool = True):
    if add_time:
        my_str = '%Y-%m-%d'
        my_str += ',%H-%M-%S'
    else:
        my_str = '%Y-%m-%d'
    return time.strftime(my_str, time.localtime())


class MovingAverageDict:
    def __init__(self, capacity, logger=None):
        self.capacity = capacity
        self.meters: dict[str, _MovingAverage] = {}
        self.logger = logger
        self.logging_method = print if logger is None else logger.info

    def __getitem__(self, key):
        if key in self.meters:
            return self.meters[key].calculate()
        msg = f"Error: You didn't define or add a value to the {key} key!"
        if self.logger is not None:
            self.logger.exception(msg)
        raise Exception(msg)

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))

            if k not in self.meters:
                self.meters[k] = _MovingAverage(self.capacity)
            self.meters[k].update(v)

    def reset_all(self):
        for meter in self.meters.values():
            meter.reset()

    def display(self, logger=None):
        for key, val in self.meters.items():
            output = f"Average {key} for the last {val.count} iterations: {val.calculate()}"
            self.logging_method(output)


class _MovingAverage:
    def __init__(self, capacity):
        self.capacity = capacity
        self.array = np.zeros(self.capacity)
        self.ind = 0
        self.count = 0
        self.sum = 0.0

    def update(self, x):
        self.sum += x - self.array[self.ind]
        self.array[self.ind] = x
        self.ind = (self.ind + 1) % self.capacity
        self.count = min(self.count + 1, self.capacity)

    # def __call__(self):
    #     return self.calculate()
    
    def calculate(self):
        return self.sum / self.count

    def reset(self):
        self.array = np.zeros(self.capacity)
        self.ind = 0
        self.count = 0
        self.sum = 0


class ETA:
    def __init__(self, total_tasks: int):
        """Estimated remaining time

        Args:
            total_tasks (int): Total tasks to be performed
        """
        self.stopwatch = Stopwatch(['total'])
        self.total_tasks = total_tasks
        
    def calculate(self, num_finished_tasks: int) -> str:
        
        result: str = self.stopwatch.calculate_estimate_remaining_time(key='total', total_tasks=self.total_tasks, num_finished_tasks=num_finished_tasks)
        
        return result
    
    def reset(self):
        self.stopwatch.reset('total')


class Stopwatch:
    """
    Stopwatch computes the time between start and stop.
    Then we can add time to the total_elapsed_time dictionary by watch name.
    """
    def __init__(self, keys: list = None):
        if keys is None:
            keys = []
        self._start_time = {k: time.time() for k in keys}

    def reset(self, key):
        self._start_time[key] = time.time()

    def elapsed_time(self, key):
        if key in self._start_time:
            return time.time() - self._start_time[key]

        self.reset(key)
        return 0.0

    @staticmethod
    def convert_to_hours_minutes(time_in_seconds: float) -> str:
        time_in_seconds = int(time_in_seconds)
        days = time_in_seconds // (24 * 3600)
        hours = (time_in_seconds % (24 * 3600)) // 3600
        minutes = (time_in_seconds % 3600) // 60
        seconds = time_in_seconds % 60

        def plural(x):
            if x != 1:
                return 's'
            return ''

        res_list = []

        if days > 0:
            res_list.append(f"{days} day{plural(days)}")
        if hours > 0:
            res_list.append(f"{hours} hour{plural(hours)}")
        if minutes > 0:
            res_list.append(f"{minutes} minute{plural(minutes)}")
        res_list.append(f"{seconds} second{plural(seconds)}")

        if len(res_list) == 1:
            return res_list[0]
        elif len(res_list) == 2:
            return ' and '.join(res_list)
        else:
            res = ', '.join(res_list[:-1]) + f', and {res_list[-1]}'
            
        return res

    def elapsed_time_in_hours_minutes(self, key):
        return self.convert_to_hours_minutes(self.elapsed_time(key))
    
    def calculate_estimate_remaining_time(self, key, total_tasks, num_finished_tasks):
        total_time = self.elapsed_time(key)
        
        if num_finished_tasks + 1 < total_tasks:
            remaining_time_str = estimated_remaining_time_string(total_time=total_time, total_tasks=total_tasks, num_finished_tasks=num_finished_tasks)
        else:
            remaining_time_str = "Elapsed time: %s" % Stopwatch.convert_to_hours_minutes(total_time)
            
        return remaining_time_str

    def __getitem__(self, name):
        return self.elapsed_time(name)

    def __getattr__(self, name: str):
        return self.elapsed_time(name)


def estimated_remaining_time_string(total_time, total_tasks, num_finished_tasks: float):
    num_finished_tasks_from_one = num_finished_tasks + 1.0
    ert = (total_tasks - num_finished_tasks_from_one) * total_time / num_finished_tasks_from_one
    res = "ETA: %s" % Stopwatch.convert_to_hours_minutes(ert)
    return res


def print_overwrite(text):
    print(" " * shutil.get_terminal_size().columns, end='\r')
    print(text, end='\r')


def set_seed(seed):
    """Sets the seed of random number generators to the predefined seed number for reproducibility.
    """
    # torch.use_deterministic_algorithms(True)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 设置Python哈希种子，为了禁止hash随机化，使得实验可复现 -> Setting a Python hash seed is necessary to prevent hash randomization and ensure the experiment's reproducibility.
    random.seed(seed)
    torch.random.manual_seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True


@torch.no_grad()
def compute_confusion_matrix(predictions: T, targets: T, num_classes: int):
    assert predictions.dim() == 1 and targets.dim() == 1
    indices = targets * num_classes + predictions
    
    conf_mat = torch.bincount(indices, minlength=num_classes ** 2)
    
    return conf_mat.reshape(num_classes, num_classes)


def calculate_accuracy(predictions: T, real_labels: T):
    return ((predictions == real_labels).float()).mean().item() * 100.0


# Borrowed from https://github.com/Naeem-Paeedeh/CPLSR
def remap_the_labels_from_zeros(labels_input: T):
    """Remaps the labels from zero to the max number of unique labels.
    """
    unique_labels_list = torch.unique(labels_input).tolist()
    assert len(unique_labels_list) > 1
    
    labels: T = torch.zeros_like(labels_input)
    for i, c in enumerate(unique_labels_list):
        labels[labels_input == c] = i
    labels = labels.long()
    
    return labels


def compute_prototypes(embeddings: T, labels: T):
    """
    embeddings_support: [num_support, embed_dim]
    labels_support: [num_support]
    """
    unique_labels = torch.unique(labels)
    num_classes = len(unique_labels)
    
    unique_labels = unique_labels - unique_labels.min()     # If should verify it does not affect the actual labelas.
    
    assert len(unique_labels) == num_classes
    
    labels_one_hot_transposed = F.one_hot(labels, num_classes=num_classes).T.to(dtype=embeddings.dtype, device=embeddings.device)
    prototypes = (labels_one_hot_transposed @ embeddings) / labels_one_hot_transposed.sum(dim=1, keepdim=True)
    return prototypes   # [num_classes, dim_embed]


# Modified from https://github.com/Naeem-Paeedeh/CPLSR

# Be careful when modifying this method because other calculations depend on it.
def cosine_similarity(features_1: T, features_2: T, only_angle: bool = True):
    # only_angle: We normalize the embeddings. Therefore, we ignore the magnitudes and only consider the angles.
    assert 1 <= len(features_1.shape) <= 2 and 1 <= len(features_2.shape) <= 2
    
    if features_1.dim() == 1:
        features_1.unsqueeze(0)
        
    if features_2.dim() == 1:
        features_2.unsqueeze(0)
    
    if only_angle:       # We consider the angle only
        features_1_norm = F.normalize(features_1, dim=-1)
        features_2_norm = F.normalize(features_2, dim=-1)
        logits_features_1_per_feature_2 = features_1_norm @ features_2_norm.t()
    else:
        logits_features_1_per_feature_2 = features_1 @ features_2.t()
    
    logits_features_2_per_feature_1 = logits_features_1_per_feature_2.t()
    return logits_features_1_per_feature_2, logits_features_2_per_feature_1


class PrototypicalClassifier(nn.Module):
    def __init__(
        self,
        distance_function: nt.DistanceFunction = nt.DistanceFunction.CosineSimilarity,
        use_learnable_tempetature: bool = True
    ):
        super().__init__()
        self.distance_function = distance_function
        self.use_learnable_tempetature = use_learnable_tempetature
        if use_learnable_tempetature:
            self.coef = nn.Parameter(torch.tensor(20.0))     # Temperature or scale
        else:
            self.coef = 20.0     # It wasn't tested!

    def calculate_loss(
        self,
        embeddings_query: T,
        prototypes: T,
        labels_query_or_probabilities: torch.LongTensor | torch.FloatTensor,
        soft_target_cross_entropy: bool = False
    ):
        """
        embeddings_query: [num_query, embed_dim]
        labels_query: [num_query]
        """

        # prototypes = self.calculate_prototypes(embeddings=embeddings_support, labels=labels_support)
        
        if not soft_target_cross_entropy:
            # Labels are remapped to the range of from zero to (number_of_classes - 1) for cross-entropy loss
            assert labels_query_or_probabilities.dtype is torch.long
            labels_query_remapped = remap_the_labels_from_zeros(labels_query_or_probabilities)
        else:       # We should use soft labels.
            assert labels_query_or_probabilities.dtype is torch.float32
        
        if self.distance_function == nt.DistanceFunction.CosineSimilarity:
            cosine_similarities, _ = cosine_similarity(embeddings_query, prototypes)   # One means the most similarity
            
            logits = self.coef * cosine_similarities
            
        elif self.distance_function == nt.DistanceFunction.CosineDistance:
            cosine_similarities, _ = cosine_similarity(embeddings_query, prototypes)   # Between -1 and 1
            
            logits = (-1.0 * self.coef) * (1.0 - cosine_similarities)       # 1.0 - cosine_similarities is between 0 and 2
        elif self.distance_function == nt.DistanceFunction.Euclidean:
            distances = torch.cdist(embeddings_query, prototypes, p=2.0)
            
            logits = (-1.0 * self.coef) * distances
        else:
            raise NotImplementedError
            
        if soft_target_cross_entropy:
            loss = F.cross_entropy(logits, labels_query_or_probabilities)
        else:
            loss = F.cross_entropy(logits, labels_query_remapped)
        
        return loss
    
    def calculate_accuracy(
        self,
        support_embeddings: T,
        support_labels: torch.LongTensor,
        query_embeddings: T,
        query_labels: torch.LongTensor,
    ):
        
        prototypes = compute_prototypes(embeddings=support_embeddings, labels=support_labels)
        
        return self.calculate_accuracy_given_prototypes(prototypes=prototypes, embeddings_query=query_embeddings, labels_query=query_labels)
        
    def calculate_accuracy_given_prototypes(
        self,
        prototypes: T,
        embeddings_query: T,
        labels_query: torch.LongTensor
    ):
        labels_query_predicted = self.predict(embeddings_query, prototypes)
        
        labels_query_remapped = remap_the_labels_from_zeros(labels_query)
        
        acc = calculate_accuracy(predictions=labels_query_predicted, real_labels=labels_query_remapped)
        
        return acc
    
    def predict(self, features: T, prototypes: T, normalize: bool = True, return_labels_or_logits=True):
        if self.distance_function == nt.DistanceFunction.CosineSimilarity:
            logits, _ = cosine_similarity(features, prototypes, only_angle=normalize)
            
            logits = self.coef * logits
            
        elif self.distance_function == nt.DistanceFunction.CosineDistance:
            logits, _ = cosine_similarity(features, prototypes, only_angle=normalize)
            
            logits = (-1.0 * self.coef) * (1.0 - logits)
            distances = -logits
        elif self.distance_function == nt.DistanceFunction.Euclidean:
            distances = torch.cdist(features, prototypes)
        elif self.distance_function == nt.DistanceFunction.L1:
            distances = torch.cdist(features, prototypes, p=1.0)
            
        if return_labels_or_logits:
            if self.distance_function == nt.DistanceFunction.CosineSimilarity:
                predictions = torch.argmax(logits, dim=-1)
            # elif self.distance_function == nt.DistanceFunction.CosineDistance:
            #     predictions = torch.argmin(distances, dim=-1)
            else:       # Distances
                predictions = torch.argmin(distances, dim=-1)
            return predictions
        else:
            if self.distance_function in [nt.DistanceFunction.Euclidean, nt.DistanceFunction.L1]:
                # logits = F.softmax(-1.0 * distances, dim=-1)
                # logits = distances
                pass
            elif self.distance_function in [nt.DistanceFunction.CosineDistance]:
                logits = F.softmax(-1.0 * distances, dim=-1)
                pass
            return logits


class EpisodeContainer:
    def __init__(self,
                 support_samples: T,
                 query_samples: T,
                 num_ways: int,
                 support_labels: T = None,
                 query_labels: T = None,
                 query_labels_one_hot: T = None,       # For logits after label-smoothing.
                 support_labels_one_hot: T = None,     # For logits after label-smoothing.
                 support_set_indices_list: list = None,
                 query_set_indices_list: list = None,
                 map_to_real_labels: T = None          # For the classification map
                 ):
        
        assert support_labels is not None or query_labels_one_hot is not None
        assert query_labels is not None or query_labels_one_hot is not None
        
        self.support_samples = support_samples
        self.query_samples = query_samples
        self.num_ways = num_ways
        
        self.support_set_indices_list: list = support_set_indices_list
        self.query_set_indices_list: list = query_set_indices_list
        self.map_to_real_labels = map_to_real_labels
        
        self.support_labels = support_labels
        self.query_labels = query_labels
        self.support_labels_one_hot = support_labels_one_hot
        self.query_labels_one_hot = query_labels_one_hot
        
        if self.support_labels is None and self.support_labels_one_hot is not None:
            self.support_labels = self.support_labels_one_hot.argmax(dim=1)
            
        if self.query_labels is None and self.query_labels_one_hot is not None:
            self.query_labels = self.query_labels_one_hot.argmax(dim=1)
        
    def move_to_device(self, device):
        self.support_samples, self.query_samples = sh.to_device([self.support_samples, self.query_samples], device=device)
        
        self.support_labels, self.query_labels = sh.to_device([self.support_labels, self.query_labels], device=device)
        
        self.support_labels_one_hot, self.query_labels_one_hot = sh.to_device([self.support_labels_one_hot, self.query_labels_one_hot], device=device)
        
    def clone(self):
        
        support_labels_clone = None
        query_labels_clone = None
        support_labels_one_hot_clone = None
        query_labels_one_hot_clone = None
        support_set_indices_list_clone = None
        query_set_indices_list_clone = None
        map_to_real_labels_clone = None
        
        if self.support_labels_one_hot is not None:
            support_labels_one_hot_clone = self.support_labels_one_hot.clone()
            
        if self.query_labels_one_hot is not None:
            query_labels_one_hot_clone = self.query_labels_one_hot.clone()
            
        if self.support_labels is not None:
            support_labels_clone = self.support_labels
        
        if self.query_labels is not None:
            query_labels_clone = self.query_labels
            
        if self.support_set_indices_list is not None:
            support_set_indices_list_clone = self.support_set_indices_list.copy()
        
        if self.query_set_indices_list is not None:
            query_set_indices_list_clone = self.query_set_indices_list.copy()
            
        if self.map_to_real_labels is not None:
            map_to_real_labels_clone = self.map_to_real_labels.copy()
            
        return EpisodeContainer(
            support_samples=self.support_samples.clone(),
            support_labels=support_labels_clone,
            query_samples=self.query_samples.clone(),
            query_labels=query_labels_clone,
            support_labels_one_hot=support_labels_one_hot_clone,
            query_labels_one_hot=query_labels_one_hot_clone,
            num_ways=self.num_ways,
            support_set_indices_list=support_set_indices_list_clone,
            query_set_indices_list=query_set_indices_list_clone,
            map_to_real_labels=map_to_real_labels_clone
        )


class EpisodeGenerator:
    def __init__(
        self,
        samples: T,
        num_shots_support: int,
        samples_test: T = None,
        labels: T = None,
        labels_test: T = None,
        labels_one_hot: T = None,
        labels_test_one_hot: T = None,
        num_shots_query: int = -1,     # If it is not set, it will use all remaining samples other than the support set as query set by default
        num_ways_limit: int = -1,      # If it is not set, it will draw the num_shots samples per class.
        device='cpu',
        dataset_name: str = '',
        keep_classes_with_insufficient_samples: bool = False,     # After pseudo-labeling or in the intermediate domain training phase, some classes might not have sufficient samples.
    ):
        self.samples = samples.cpu()
        self.num_shots_support = num_shots_support
        self.num_shots_query = num_shots_query
        self.device = device
        self.dataset_name = dataset_name
        self.are_train_and_test_sets_separated = self.dataset_name == 'Houston'
        self.keep_classes_with_insufficient_samples = keep_classes_with_insufficient_samples
        
        self.samples_test = samples_test
        self.labels: T = labels
        self.labels_test: T = labels_test
        self.labels_one_hot: T = labels_one_hot
        self.labels_test_one_hot: T = labels_test_one_hot
        
        self.num_ways_limit = num_ways_limit
        
        self.labels_are_integers: bool = labels is not None
        self.num_ways_max = -1
        
        self._transfer_data_to_cpu()
        
        self.label_to_indices = {}
        self.label_to_indices_test = {}     # For the Houston dataset.
        
        self.useful_labels = self._find_classes_with_sufficient_samples()
        
        if self.labels_are_integers:
            self.num_ways_max = len(self.useful_labels)
            
        if self.num_ways_limit == -1:
            self.num_ways_limit = self.num_ways_max
        self.num_ways_effective = min(self.num_ways_limit, self.num_ways_max)
        
        assert self.num_ways_effective > 1
        
    def generate(self, keep_permuted_indices: bool = False, calculate_one_hot_labels: bool = True) -> sh.EpisodeContainer:
        """It generates the support and query sets.

        Args:
            keep_permuted_indices (bool, optional): For classificatin map. Defaults to False.

        Returns:
            EpisodeContainer: _description_
        """
        if self.labels_are_integers:
            classes_permuted = torch.randperm(self.num_ways_max)
            chosen_classes = self.useful_labels[classes_permuted][:self.num_ways_effective].tolist()
        else:
            chosen_classes = self.useful_labels.tolist()
        
        support_set_indices_list = []
        query_set_indices_list = []
        
        for lbl, lbl_real in enumerate(chosen_classes):
            indices = self.label_to_indices[lbl_real]
            perm = torch.randperm(len(indices))
            indices_permuted = indices[perm]
            
            if self.are_train_and_test_sets_separated:      # For the Houston dataset
                indices_test = self.label_to_indices_test[lbl_real]
                perm_test = torch.randperm(len(indices_test))
                indices_test_permuted = indices_test[perm_test]
            
            support_set_indices_list += indices_permuted[:self.num_shots_support].tolist()
            
            if self.are_train_and_test_sets_separated:
                if self.num_shots_query > 0:
                    query_set_indices_list += indices_test_permuted[:self.num_shots_query].tolist()
                elif self.num_shots_query == -1:
                    # Following the Gia-CFSL, FDFSL, and CDFS-CASCL implementations, we use the remaining samples as the query set (test set). See their get_train_test_loader method.
                    query_set_indices_list += indices_test_permuted.tolist()
                else:
                    raise NotImplementedError
            else:
                if self.num_shots_query > 0:
                    query_set_indices_list += indices_permuted[self.num_shots_support:self.num_shots_support + self.num_shots_query].tolist()
                # Following the Gia-CFSL, FDFSL, and CDFS-CASCL implementations, we use the remaining samples as the query set (test set). See their get_train_test_loader method.
                elif self.num_shots_query == -1:
                    query_set_indices_list += indices_permuted[self.num_shots_support:].tolist()
                else:
                    raise NotImplementedError
                
            pass
            
        support_samples = self.samples[support_set_indices_list]
        
        support_labels_not_mapped = self.labels[support_set_indices_list]
        
        if self.are_train_and_test_sets_separated:
            query_samples = self.samples_test[query_set_indices_list]
            query_labels_not_mapped = self.labels_test[query_set_indices_list]
        else:
            query_samples = self.samples[query_set_indices_list]
            query_labels_not_mapped = self.labels[query_set_indices_list]
        
        # We remap the labels to be in range [0, self.num_ways_effective].
        support_labels: T = torch.zeros_like(support_labels_not_mapped)
        query_labels: T = torch.zeros_like(query_labels_not_mapped)
        
        # Background is zero.
        map_to_real_labels = torch.zeros(len(chosen_classes) + 1, dtype=torch.long)
        
        for label_new, label_real in enumerate(chosen_classes):
            support_labels[support_labels_not_mapped == label_real] = label_new
            query_labels[query_labels_not_mapped == label_real] = label_new
            map_to_real_labels[label_new + 1] = label_real + 1  # Background is zero.
        
        support_samples, support_labels = sh.to_device([support_samples, support_labels], device=self.device)
        
        query_samples, query_labels = sh.to_device([query_samples, query_labels], device=self.device)
        
        support_labels_one_hot = None
        query_labels_one_hot = None
        
        if calculate_one_hot_labels:
            # if return_logits:
            if self.labels_are_integers:
                support_labels_one_hot = F.one_hot(support_labels, num_classes=self.num_ways_effective)
                query_labels_one_hot = F.one_hot(query_labels, num_classes=self.num_ways_effective)
            else:
                support_labels_one_hot = self.labels_one_hot[support_set_indices_list]
                query_labels_one_hot = self.labels_one_hot[query_set_indices_list]
                
                support_labels_one_hot, query_labels_one_hot = sh.to_device([support_labels_one_hot, query_labels_one_hot], device=self.device)
            
        episode = sh.EpisodeContainer(
            support_samples=support_samples,
            support_labels=support_labels,
            query_samples=query_samples,
            query_labels=query_labels,
            num_ways=self.num_ways_effective,
            support_labels_one_hot=support_labels_one_hot,
            query_labels_one_hot=query_labels_one_hot,
            support_set_indices_list=support_set_indices_list if keep_permuted_indices else None,
            query_set_indices_list=query_set_indices_list if keep_permuted_indices else None,
            map_to_real_labels=map_to_real_labels
        )
        
        return episode
    
    def _transfer_data_to_cpu(self):
        # We should give the network either the classes as integers or one-hot labels.
        assert self.labels is not None or self.labels_one_hot is not None
        assert self.labels is None or self.labels_one_hot is None
        
        if self.are_train_and_test_sets_separated:
            self.samples_test = self.samples_test.cpu()
            assert self.labels_test is not None or self.labels_test_one_hot is not None
            assert self.labels_test is None or self.labels_test_one_hot is None
            assert self.labels_test.dim() == 1 and self.labels_test.dtype == torch.long
            self.labels_test = self.labels_test.cpu()
            
        if self.labels is not None:
            assert self.labels.dim() == 1 and self.labels.dtype == torch.long
            self.labels = self.labels.cpu()
        
        if self.labels_one_hot is not None:
            assert self.labels_one_hot.dim() == 2        # When they are logit vectors (pseudo-labels after label-smoothing)
            if self.labels is None:
                self.labels = torch.argmax(self.labels_one_hot, dim=-1)
            else:
                NotImplementedError
            assert self.num_ways_limit == -1 or self.num_ways_limit == self.labels_one_hot.shape[1]
            self.num_ways_max = self.labels_one_hot.shape[1]
            self.labels_one_hot = self.labels_one_hot.cpu()
        
        if self.labels_test_one_hot is not None:
            assert self.labels_test_one_hot.dim() == 2        # When they are logit vectors (pseudo-labels after label-smoothing)
            if self.labels_test is None:
                self.labels_test = torch.argmax(self.labels_test_one_hot, dim=-1)
            else:
                NotImplementedError
            self.labels_test_one_hot = self.labels_test_one_hot.cpu()
            
        if self.are_train_and_test_sets_separated:
            len(self.labels.unique()) == len(self.labels_test.unique())
        
    def _find_classes_with_sufficient_samples(self):
        # Verifying whether there are sufficient samples in every class or not
        # assert self.num_shots_query > 0 or self.num_shots_query == -1
        
        suffix_str = f' of the "{self.dataset_name}" dataset'
        
        classes_with_sufficient_samples = []
        
        for lbl in self.labels.unique().tolist():
            self.label_to_indices[lbl] = torch.nonzero(self.labels == lbl).squeeze()
            
            if self.are_train_and_test_sets_separated:
                self.label_to_indices_test[lbl] = torch.nonzero(self.labels_test == lbl).squeeze()
            
            if len(self.label_to_indices[lbl]) < self.num_shots_support:
                msg = "There aren't sufficient"
                if self.keep_classes_with_insufficient_samples:
                    logging.warning(f'Warning: {msg}')
                else:
                    raise Exception(f'Error: {msg}')
            
            if not self.keep_classes_with_insufficient_samples:
                flag_insufficient_samples = True
                
                if self.are_train_and_test_sets_separated:
                    # It has separate ground truth labels for the train and test sets.
                    if self.num_shots_query == -1 and len(self.label_to_indices[lbl]) > 1:      # We require at least two samples.
                        flag_insufficient_samples = False
                    elif self.num_shots_query > 0 and len(self.label_to_indices[lbl]) >= self.num_shots_support and len(self.label_to_indices_test[lbl]) >= self.num_shots_query:
                        flag_insufficient_samples = False
                    else:
                        raise NotImplementedError
                else:
                    # They only have one ground truth labels set.
                    if self.num_shots_query == -1 and len(self.label_to_indices[lbl]) > 1:      # We require at least two samples.
                        flag_insufficient_samples = False
                    elif self.num_shots_query > 0 and len(self.label_to_indices[lbl]) >= self.num_shots_support + self.num_shots_query:
                        flag_insufficient_samples = False
                    else:
                        raise NotImplementedError
            else:
                flag_insufficient_samples = False
            
            if flag_insufficient_samples:
                msg = f"There are not sufficient samples for class {lbl}{suffix_str}."
                if not self.keep_classes_with_insufficient_samples:
                    logging.error('Error:' + msg)
                    raise Exception(msg)
                else:
                    logging.warning('Warning:' + msg)
            classes_with_sufficient_samples.append(lbl)
            
        return torch.tensor(classes_with_sufficient_samples)
    

# This is equivalent to the scipy.stats.wasserstein_distance_nd
@torch.no_grad()
def W_distance(x, y):
    # x, y: [n, d]
    # Calculate cost matrix (Euclidean distance)
    M = torch.cdist(x, y, p=2)

    # Uniform weights
    a = torch.ones(x.shape[0], device=x.device) / x.shape[0]
    b = torch.ones(y.shape[0], device=y.device) / y.shape[0]

    # Exact EMD
    return ot.emd2(a, b, M)


def highlighted_message(message: str, max_length: int = 60):
    assert len(message) < max_length - 4
    margin = max_length - 4 - len(message) // 2   # :)
    result = '-' * margin + ' ' + message + ' ' + '-' * margin
    return result