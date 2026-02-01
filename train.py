#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, TensorDataset

import numpy as np
import os
import argparse

from sklearn import metrics

import utils as utils
from MIFOMO import MIFOMO, MixupScheduler
from torch import Tensor as T
import new_types as nt
import shared as sh
from collections import defaultdict
from lgc import LGC
import logging


# https://github.com/Naeem-Paeedeh/ADAPTER
def concatenate_features(features, n_times):
    lst = []
    for _ in range(n_times):
        lst.append(features)
    return torch.cat(lst, dim=-1)


@torch.no_grad()
def refine_one_hot_pseuodo_labels(
    support_samples: T,
    support_labels_one_hot: T,
    query_samples: T,
    query_pseuodo_labels_one_hot: T,
    minimum_requires_samples: int,
    top_k: int = -1
):
    """This method removes the classes that do not have any samples.
    """
    
    # Step 1: We find keep the classes with sufficient samples.
    assert query_pseuodo_labels_one_hot.dim() == 2
    predicted_query_labels = query_pseuodo_labels_one_hot.argmax(dim=1)
    
    num_samples_per_class = torch.bincount(predicted_query_labels)
    chosen_classes_list = torch.nonzero(num_samples_per_class >= minimum_requires_samples).squeeze().tolist()
    
    refined_query_labels_one_hot1 = query_pseuodo_labels_one_hot[:, chosen_classes_list]
    refined_support_labels_one_hot_final = support_labels_one_hot[:, chosen_classes_list]
    
    # Step 2: We keep the samples with top-k highest confidence scores.
    if top_k > 0:
        predicted_query_labels = refined_query_labels_one_hot1.argmax(dim=1)
        unique_classes_list = predicted_query_labels.unique().tolist()
        confidence_scores = refined_query_labels_one_hot1.max(dim=1)[0]
        
        indices_sorted_by_scores = torch.argsort(confidence_scores, descending=True, stable=True)
        inferred_labels_sorted_by_scores = predicted_query_labels[indices_sorted_by_scores]
        
        chosen_classes_list = []
        
        for lbl in unique_classes_list:
            mask_labels_sorted_by_scores_for_current_class = inferred_labels_sorted_by_scores == lbl
            indices_of_indices_belong_to_this_class = torch.nonzero(mask_labels_sorted_by_scores_for_current_class).squeeze()
            # assert len(indices_of_indices_belong_to_this_class) >= top_k
            if len(indices_of_indices_belong_to_this_class) < top_k:
                logging.warning(f'Warning: Class {lbl} does not have sufficient samples to choose top-k={top_k} for refinement!')
            
            chosen_indices_of_indices_belong_to_this_class = indices_of_indices_belong_to_this_class[:top_k].tolist()
            
            chosen_indices_final = indices_sorted_by_scores[chosen_indices_of_indices_belong_to_this_class]
            
            chosen_classes_list += chosen_indices_final.tolist()
            
    refined_query_samples_final = query_samples[chosen_classes_list]
    refined_query_labels_one_hot_final = refined_query_labels_one_hot1[chosen_classes_list]
    
    # After filtering the classes, some samples may have a zero logit vector.
    mask_query_indices_of_samples_to_keep = refined_query_labels_one_hot_final.sum(dim=1) > 0
    refined_query_samples_final = refined_query_samples_final[mask_query_indices_of_samples_to_keep]
    refined_query_labels_one_hot_final = refined_query_labels_one_hot_final[mask_query_indices_of_samples_to_keep]
    
    mask_support_indices_of_samples_to_keep = refined_support_labels_one_hot_final.sum(dim=1) > 0
    refined_support_samples_final = support_samples[mask_support_indices_of_samples_to_keep]
    refined_support_labels_one_hot_final = refined_support_labels_one_hot_final[mask_support_indices_of_samples_to_keep]
    
    episode_result = sh.EpisodeContainer(
        support_samples=refined_support_samples_final,
        support_labels_one_hot=refined_support_labels_one_hot_final,
        query_samples=refined_query_samples_final,
        query_labels_one_hot=refined_query_labels_one_hot_final,
        num_ways=refined_query_labels_one_hot_final.shape[1]
    )
    
    return episode_result


class Stages:
    def __init__(
        self,
        settings_file: str
    ):
        self.settings_file = settings_file
        self.mifomo = MIFOMO(settings_file)
        self.device = self.mifomo.device
        
    def source_phase(self):
        ma = sh.MovingAverageDict(20)
        
        num_episodes = self.mifomo.num_episodes_source
        
        eta = sh.ETA(total_tasks=num_episodes)
        
        self.mifomo.train()
        
        optimizer = self.mifomo.obtain_the_optimizer()
        
        for ep_num in range(num_episodes):
            episode = self.mifomo.generate_an_episode(source_or_target=True)
            
            embeddings_support = self.mifomo.forward(episode.support_samples, source_or_target=True)
            
            embeddings_query = self.mifomo.forward(episode.query_samples, source_or_target=True)
            
            prototypes = sh.compute_prototypes(embeddings_support, episode.support_labels)
            
            # assert prototypes.shape[0] == mifomo.num_ways_training
            
            loss_mx = 0.0
            
            if self.mifomo.use_mixup:
                num_samples_query = embeddings_query.shape[0]
                # Sampling with replacement
                size = (self.mifomo.num_mixup_source,)
                indices_1 = torch.randint(0, num_samples_query, size)
                indices_2 = torch.randint(0, num_samples_query, size)
            
                lmbda_tensor = self.mifomo.beta_distribution.sample(size).to(self.device).unsqueeze(1)
            
                embeddings_mixed = lmbda_tensor * embeddings_query[indices_1] + (1.0 - lmbda_tensor) * embeddings_query[indices_2]
            
                query_labels_one_hot = episode.query_labels_one_hot
                labels_mixed = lmbda_tensor * query_labels_one_hot[indices_1] + (1.0 - lmbda_tensor) * query_labels_one_hot[indices_2]
            
                loss_mx = self.mifomo.prototypical_classifier.calculate_loss(embeddings_query=embeddings_mixed, prototypes=prototypes, labels_query_or_probabilities=labels_mixed, soft_target_cross_entropy=True)
            
            loss_fsl = self.mifomo.prototypical_classifier.calculate_loss(embeddings_query=embeddings_query, prototypes=prototypes, labels_query_or_probabilities=episode.query_labels)
            
            acc_fsl = self.mifomo.prototypical_classifier.calculate_accuracy_given_prototypes(prototypes=prototypes, embeddings_query=embeddings_query, labels_query=episode.query_labels)
            
            ma.update(acc=acc_fsl)
            
            loss = loss_fsl + loss_mx
            
            ma.update(loss=loss)
            
            remaining_time_str = eta.calculate(num_finished_tasks=ep_num)
            loss_str = f'Loss: {ma['loss']: .5f}'
            report = f'Source domain, Episode: {ep_num + 1}/{num_episodes}, loss: {loss_str}, acc: {ma['acc']: .2f}, {remaining_time_str}'
            sh.print_overwrite(report)
            if ep_num + 1 % 500 == 0:
                logging.info(report)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if ep_num + 1 % 1000 == 0:
                self.mifomo.save('model_phase_1', acc=ma['acc'], loss=ma['loss'], num_episodes=ep_num)
        
        logging.info(report)
        self.mifomo.save('model_phase_1', acc=ma['acc'], loss=ma['loss'], num_episodes=num_episodes)
        
    def _evaluate_save_in_dict_and_report_accuracy(
        self,
        acc_dict: dict,
        desc_and_key: str,
        episode_target: sh.EpisodeContainer,
        episode_num: int,
        num_episodes: int,
        show_arrow: bool = False
    ):
        acc_without_label_smoothing, acc_with_label_smoothing, A_without_label_smoothing, A_with_label_smoothing, kappa_without_label_smoothing, kappa_with_label_smoothing = self._evaluate_episode(episode=episode_target)
        
        if desc_and_key not in acc_dict:
            acc_dict[desc_and_key] = defaultdict(list)
        
        episode_str = f"Episode: {episode_num + 1}/{num_episodes},"
        
        acc_dict[desc_and_key]['acc_without_LS'].append(acc_without_label_smoothing)
        acc_dict[desc_and_key]['A_without_label_smoothing'].append(A_without_label_smoothing)
        acc_dict[desc_and_key]['kappa_without_label_smoothing'].append(kappa_without_label_smoothing)
        
        logging.info(f'{episode_str} {desc_and_key}, w/o label smoothing: {acc_without_label_smoothing:.2f}')
        
        suffix = ''
        if show_arrow:
            suffix = '\t<--'
        
        if self.mifomo.use_label_smoothing:
            acc_dict[desc_and_key]['acc_with_LS'].append(acc_with_label_smoothing)
            acc_dict[desc_and_key]['A_with_label_smoothing'].append(A_with_label_smoothing)
            acc_dict[desc_and_key]['kappa_with_label_smoothing'].append(kappa_with_label_smoothing)
            logging.info(f'{episode_str} {desc_and_key}, with label smoothing: {acc_with_label_smoothing:.2f}{suffix}')
    
    def _report_accuracies(
        self,
        acc_dict: dict,
        stages_keys_list: list,
        eta: sh.ETA,
        episode_num: int,
        num_episodes_tests: int,
    ):
        # Reporting the average accuracies at each step:
        
        logging.info('-' * 70)
        
        for i, stage_key in enumerate(stages_keys_list):
            
            if stage_key not in acc_dict.keys():
                continue
            
            suffix = ''
            
            suffix = f', {eta.calculate(num_finished_tasks=episode_num)}'
                
            suffix2 = ''
            if i in {0, 2, len(stages_keys_list)}:
                suffix2 = '\t<--'
                
            episode_str = f"Episode: {episode_num + 1}/{num_episodes_tests},"
            
            acc_without_LS = acc_dict[stage_key]['acc_without_LS']
            OAMean = np.mean(acc_without_LS)
            OAStd = np.std(acc_without_LS)
            logging.info(f'{episode_str} Avg. OA, {stage_key}, w/o LS: {OAMean:.2f} ± {OAStd:.2f}{suffix}')
            
            A_without_label_smoothing = acc_dict[stage_key]['A_without_label_smoothing']
            kappa_without_label_smoothing = acc_dict[stage_key]['kappa_without_label_smoothing']
            AA = np.mean(A_without_label_smoothing, 1)
            AAMean = np.mean(AA, 0)
            AAStd = np.std(AA)
            logging.info(f"{episode_str} Avg. AA, {stage_key} w/o LS: {100 * AAMean:.2f} ± {100 * AAStd:.2f}")      # 100 * AAStd?
            kMean = np.mean(kappa_without_label_smoothing)
            kStd = np.std(kappa_without_label_smoothing)
            logging.info(f"{episode_str} Avg. kappa, {stage_key} w/o LS: {100 * kMean:.2f} ± {100 * kStd:.2f}")
            
            if self.mifomo.use_label_smoothing:
                acc_with_LS = acc_dict[stage_key]['acc_with_LS']
                OAMean = np.mean(acc_with_LS)
                OAStd = np.std(acc_with_LS)
                logging.info(f'{episode_str} Avg. OA, {stage_key}, with LS: {OAMean:.2f} ± {OAStd:.2f}{suffix}{suffix2}')
            
                A_with_label_smoothing = acc_dict[stage_key]['A_with_label_smoothing']
                AA = np.mean(A_with_label_smoothing, 1)
                AAMean = np.mean(AA, 0)
                AAStd = np.std(AA)
                logging.info(f"{episode_str} Avg. AA, {stage_key} with LS: {100 * AAMean:.2f} ± {100 * AAStd:.2f}")

                kappa_with_label_smoothing = acc_dict[stage_key]['kappa_with_label_smoothing']
                kMean = np.mean(kappa_with_label_smoothing)
                kStd = np.std(kappa_with_label_smoothing)
                logging.info(f"{episode_str} Avg. kappa, {stage_key} with LS: {100 * kMean:.2f} ± {100 * kStd:.2f}")

                AMean = np.mean(A_with_label_smoothing, 0)
                AStd = np.std(A_with_label_smoothing, 0)
            else:
                AMean = np.mean(A_without_label_smoothing, 0)
                AStd = np.std(A_without_label_smoothing, 0)

            if i == 5 or i == len(stages_keys_list) - 1 or stage_key == 'acc_after_intermediate_domain_training':
                logging.info(f"{episode_str} Accuracy for each class:")
                for i in range(self.mifomo.num_classes_target):
                    logging.info(f"Class {i + 1}: {100 * AMean[i]:.2f} ± {100 * AStd[i]:.2f}")
        
        logging.info('*' * 70)
    
    def intermediate_phase(
        self,
    ):
        optimizer = self.mifomo.obtain_the_optimizer()
        acc_dict = {}
        
        stages_keys_list = [
            'acc_after_source_domain',
            'acc_after_train_with_the_target_support_set',
            'acc_after_intermediate_domain_training',
            'acc_after_complete_training'
        ]
        
        num_episodes_tests = self.mifomo.num_episodes_tests       # 10 # nDataSet in other codes
        
        eta = sh.ETA(total_tasks=num_episodes_tests)
        
        LS_status = ''
        if not self.mifomo.use_label_smoothing:
            LS_status = ',WO_LS'
            
        episode_target: sh.EpisodeContainer = None
        
        for ep_num in range(num_episodes_tests):
            self.mifomo.reset_the_model()
            
            sh.set_seed(self.mifomo.seed + ep_num)
            
            episode_target = self.mifomo.generate_an_episode(source_or_target=False)
            self._evaluate_save_in_dict_and_report_accuracy(acc_dict=acc_dict, desc_and_key=stages_keys_list[0], episode_target=episode_target, episode_num=ep_num, num_episodes=num_episodes_tests)
            
            # We train the network on the target domain with the support set to be able to assign pseudo-labels to the query set samples.
            episode_str_for_cache_file = f'DS={self.mifomo.dataset_name_target},n_ep_target1={self.mifomo.num_episodes_target1}'
            
            if self.mifomo.perform_the_intermediate_domain_training:
                self.prepare_for_pseudo_labeling(optimizer=optimizer, episode_target=episode_target, num_episodes=self.mifomo.num_episodes_target1, cache_file=f'step1,{episode_str_for_cache_file}{LS_status},episode={ep_num}.pth')
                
                self._evaluate_save_in_dict_and_report_accuracy(acc_dict=acc_dict, desc_and_key=stages_keys_list[1], episode_target=episode_target, show_arrow=True, episode_num=ep_num, num_episodes=num_episodes_tests)
            
                # We assign pseudo-labels to the query set samples of the target domain.
                # if mifomo.use_label_smoothing:
                # query_predicted.shape: [num_samples, num_ways]
                query_target_predicted_without_LS, query_target_predicted_with_LS = self.predict_query_labels(episode=episode_target)
                
                if self.mifomo.use_label_smoothing:
                    query_target_predicted = query_target_predicted_with_LS
                else:
                    query_target_predicted = query_target_predicted_without_LS
                
                episode_target_pseudo_labeled = refine_one_hot_pseuodo_labels(
                    support_samples=episode_target.support_samples,
                    support_labels_one_hot=episode_target.support_labels_one_hot,
                    query_samples=episode_target.query_samples,
                    query_pseuodo_labels_one_hot=query_target_predicted,
                    minimum_requires_samples=self.mifomo.num_shots_support_intermediate_domain + self.mifomo.num_shots_query_intermediate_domain,
                    top_k=self.mifomo.top_k_most_confident
                )
                
                episode_str_for_cache_file += f',n_ep_intrm={self.mifomo.num_episodes_intermediate_domain}'
                
                self.train_intermediate_domain(
                    episode_target_pseudo_labeled=episode_target_pseudo_labeled,
                    optimizer=optimizer,
                    num_episodes=self.mifomo.num_episodes_intermediate_domain,
                    cache_file=f'step2,{episode_str_for_cache_file}{LS_status},episode={ep_num}.pth'
                    # cache_file=None
                )
                    
                self._evaluate_save_in_dict_and_report_accuracy(acc_dict=acc_dict, desc_and_key=stages_keys_list[2], episode_target=episode_target, episode_num=ep_num, num_episodes=num_episodes_tests)
               
            self._report_accuracies(
                acc_dict=acc_dict,
                stages_keys_list=stages_keys_list,
                eta=eta,
                episode_num=ep_num,
                num_episodes_tests=num_episodes_tests
            )
        
        return episode_target
        
    def train_intermediate_domain(
        self,
        episode_target_pseudo_labeled: sh.EpisodeContainer,
        optimizer,
        num_episodes: int,
        cache_file: str = None,
    ):
        if self.mifomo.try_cache_first(cache_file):
            return
        
        device = self.device
        
        self.mifomo.train()
        
        num_classes_two_domains = self.mifomo.num_classes_source + episode_target_pseudo_labeled.num_ways
        
        # Note the num_ways should be the total classes from the source and target domains.
        source_EpisodeGenerator = sh.EpisodeGenerator(
            samples=self.mifomo.samples_source,
            labels_one_hot=F.one_hot(self.mifomo.labels_source, num_classes_two_domains),
            num_shots_support=self.mifomo.num_shots_support_intermediate_domain,
            num_shots_query=self.mifomo.num_shots_query_intermediate_domain,
            num_ways_limit=-1,
            dataset_name=self.mifomo.dataset_name_source,
            device='cpu',
        )
        
        all_target_labels_one_hot = torch.cat([
            episode_target_pseudo_labeled.support_labels_one_hot.cpu(),
            episode_target_pseudo_labeled.query_labels_one_hot.cpu()        # These are the query pseudo-labels after refinement.
        ])
        
        all_target_labels_one_hot_extended = torch.cat(
            [
                torch.zeros(all_target_labels_one_hot.shape[0], self.mifomo.num_classes_source),
                all_target_labels_one_hot,
            ],
            dim=1,
        )
        
        target_EpisodeGenerator = sh.EpisodeGenerator(
            samples=torch.cat([episode_target_pseudo_labeled.support_samples.cpu(), episode_target_pseudo_labeled.query_samples.cpu()]),
            labels_one_hot=all_target_labels_one_hot_extended,                      # We use the pseudo-labels.
            num_shots_support=self.mifomo.num_shots_support_intermediate_domain,    # num_shots_support_divided
            num_shots_query=self.mifomo.num_shots_query_intermediate_domain,
            num_ways_limit=-1,
            device='cpu',
            dataset_name='Pseudo-labeled query samples in the intermediate domain.',
            keep_classes_with_insufficient_samples=True
        )
        
        ma = sh.MovingAverageDict(20)
        eta = sh.ETA(total_tasks=num_episodes)
        ms = MixupScheduler(total_iterations=num_episodes, temperature=self.mifomo.temperature_mixup_scheduler)
        
        for ep_num in range(num_episodes):
            episode_source = source_EpisodeGenerator.generate()
            episode_target = target_EpisodeGenerator.generate()
            
            embeddings_support_source = self.mifomo.forward(episode_source.support_samples, source_or_target=True)
            embeddings_support_target = self.mifomo.forward(episode_target.support_samples, source_or_target=False)
            
            prototypes_source = sh.compute_prototypes(embeddings_support_source, episode_source.support_labels)
            prototypes_target = sh.compute_prototypes(embeddings_support_target, episode_target.support_labels)
            
            prototypes_both_domains = torch.cat([prototypes_source, prototypes_target])
            
            embeddings_query_source = self.mifomo.forward(episode_source.query_samples, source_or_target=True)
            embeddings_query_target = self.mifomo.forward(episode_target.query_samples, source_or_target=False)
            
            # Sampling with replacement
            size: tuple[int] = (self.mifomo.num_mixup_intermediate,)
            indices_source = torch.randint(0, embeddings_query_source.shape[0], size).tolist()
            indices_target = torch.randint(0, embeddings_query_target.shape[0], size).tolist()
            
            lmbda_tensor = ms.sample_lambda(num_samples=self.mifomo.num_mixup_intermediate).view(-1, 1).to(self.device)
            
            embeddings_mixed = lmbda_tensor * embeddings_query_source[indices_source] + (1.0 - lmbda_tensor) * embeddings_query_target[indices_target]
            
            labels_one_hot_mixed = lmbda_tensor * episode_source.query_labels_one_hot[indices_source].to(device) + (1.0 - lmbda_tensor) * episode_target.query_labels_one_hot[indices_target].to(device)
            
            labels_one_hot_mixed = labels_one_hot_mixed
            
            # Uniform sampling
            lmbda_tensor = lmbda_tensor.view(-1, 1, 1, 1)
            mixed_inputs = lmbda_tensor * episode_source.query_samples[indices_source].to(device) + (1.0 - lmbda_tensor) * episode_target.query_samples[indices_target].to(device)
            
            embeddings_mixed_inputs = self.mifomo.forward(mixed_inputs, source_or_target=False)
            
            loss1 = self.mifomo.prototypical_classifier.calculate_loss(embeddings_query=embeddings_mixed_inputs, prototypes=prototypes_both_domains, labels_query_or_probabilities=labels_one_hot_mixed, soft_target_cross_entropy=True)
            
            loss2 = self.mifomo.prototypical_classifier.calculate_loss(embeddings_query=embeddings_mixed, prototypes=prototypes_both_domains, labels_query_or_probabilities=labels_one_hot_mixed, soft_target_cross_entropy=True)
            
            loss = loss1 + loss2
            
            ma.update(loss=loss)
            
            remaining_time_str = eta.calculate(num_finished_tasks=ep_num)
            loss_str = f'Loss: {ma['loss']: .5f}'
            report = f'Intermediate domain, Episode: {ep_num + 1}/{num_episodes}, loss: {loss_str}, {remaining_time_str}'
            sh.print_overwrite(report)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            ms.update(
                step=ep_num,
                embeddings_source=embeddings_query_source,
                embeddings_target=embeddings_query_target,
                embeddings_intermediate=embeddings_mixed
            )
            
        self.mifomo.save(file_name=cache_file, cache=True)
        
    def prepare_for_pseudo_labeling(
        self,
        optimizer,
        episode_target: sh.EpisodeContainer,
        num_episodes: int,
        cache_file: str = None
    ):
        
        if self.mifomo.try_cache_first(cache_file):
            return
        
        ma = sh.MovingAverageDict(10)
        
        eta = sh.ETA(total_tasks=num_episodes)
        
        self.mifomo.train()
        
        optimizer = self.mifomo.obtain_the_optimizer()
        
        for ep_num in range(num_episodes):
            episode_new = self.divide_support_set_to_support_and_query_sets(episode_target=episode_target)
            
            embeddings_new_support = self.mifomo.forward(episode_new.support_samples, source_or_target=False)
            
            embeddings_new_query = self.mifomo.forward(episode_new.query_samples, source_or_target=False)
            
            prototypes = sh.compute_prototypes(embeddings_new_support, episode_new.support_labels)
            
            loss_mx = 0
            
            if self.mifomo.use_mixup:
                # Sampling with replacement
                num_samples_query = embeddings_new_query.shape[0]
                size = (self.mifomo.num_mixup_target,)
                indices_1 = torch.randint(0, num_samples_query, size)
                indices_2 = torch.randint(0, num_samples_query, size)
            
                lmbda_tensor = self.mifomo.beta_distribution.sample(size).to(self.device).unsqueeze(1)
            
                embeddings_mixed = lmbda_tensor * embeddings_new_query[indices_1] + (1.0 - lmbda_tensor) * embeddings_new_query[indices_2]
            
                query_labels_one_hot = episode_new.query_labels_one_hot
                labels_mixed = lmbda_tensor * query_labels_one_hot[indices_1] + (1.0 - lmbda_tensor) * query_labels_one_hot[indices_2]
            
                loss_mx = self.mifomo.prototypical_classifier.calculate_loss(embeddings_query=embeddings_mixed, prototypes=prototypes, labels_query_or_probabilities=labels_mixed, soft_target_cross_entropy=True)
                
            loss_fsl = self.mifomo.prototypical_classifier.calculate_loss(embeddings_query=embeddings_new_query, prototypes=prototypes, labels_query_or_probabilities=episode_new.query_labels, soft_target_cross_entropy=False)
            
            acc_fsl = self.mifomo.prototypical_classifier.calculate_accuracy_given_prototypes(prototypes=prototypes, embeddings_query=embeddings_new_query, labels_query=episode_new.query_labels)
                
            ma.update(acc=acc_fsl)
            
            loss = loss_fsl + loss_mx
            
            ma.update(loss=loss)
            
            remaining_time_str = eta.calculate(num_finished_tasks=ep_num)
            loss_str = f'Loss: {ma['loss']: .5f}'
            report = f'Target domain (support set),Episode: {ep_num + 1}/{num_episodes}, loss: {loss_str}, acc: {ma['acc']: .2f}, {remaining_time_str}'
            sh.print_overwrite(report)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        logging.info(report)
        
        self.mifomo.save(file_name=cache_file, cache=True)
    
    @torch.no_grad()
    def predict_query_labels(
        self,
        episode: sh.EpisodeContainer,
        source_or_target: bool = False
    ):
        self.mifomo.eval()
        
        support_samples = episode.support_samples
        support_labels = episode.support_labels
        query_samples = episode.query_samples
        # query_labels = episode.query_labels
        
        self.mifomo.eval()
        
        support_embeddings = self.mifomo.forward(support_samples, source_or_target=source_or_target).cpu()
        support_labels = support_labels.cpu()
        
        support_logits = F.one_hot(support_labels, num_classes=episode.num_ways).float()
        
        prototypes = sh.compute_prototypes(embeddings=support_embeddings, labels=support_labels)
        
        dataset = TensorDataset(query_samples)
        dataloader = DataLoader(dataset, batch_size=self.mifomo.batch_size_LGC, drop_last=False)
        
        logits_all_before_label_smoothing = torch.Tensor()
        logits_all_after_label_smoothing = None
        
        if self.mifomo.use_label_smoothing:
            lgc = LGC(distance_function=self.mifomo.distance_function)
            logits_all_after_label_smoothing = torch.Tensor()
            
        for x, in dataloader:
            query_embeddings = self.mifomo.forward(x, source_or_target=source_or_target).cpu()
        
            query_logits = self.mifomo.prototypical_classifier.predict(features=query_embeddings, prototypes=prototypes, return_labels_or_logits=False)      # Its shape: [num_samples, num_ways]
            
            logits_all_before_label_smoothing = torch.cat([logits_all_before_label_smoothing, query_logits])
            
            if self.mifomo.use_label_smoothing:
                logits_after_label_smoothing = lgc.compute(x=torch.cat([support_embeddings, query_embeddings]), y_bar=torch.cat([support_logits, query_logits]))
                logits_after_label_smoothing_after_discarding_the_support_set = logits_after_label_smoothing[support_samples.shape[0]:]
                logits_all_after_label_smoothing = torch.cat([logits_all_after_label_smoothing, logits_after_label_smoothing_after_discarding_the_support_set])
            
        if self.mifomo.use_label_smoothing:
            logits_all_after_label_smoothing = logits_all_after_label_smoothing.detach()
        
        return logits_all_before_label_smoothing.detach(), logits_all_after_label_smoothing   # Its shape: [num_samples, num_ways]
    
    def divide_support_set_to_support_and_query_sets(
        self,
        episode_target: sh.EpisodeContainer
    ):
        
        num_ways = episode_target.num_ways
        
        assert episode_target.num_ways >= 4        # 5 classes -> 3 classes for support, 2 classes for query  # 4 classes -> 2 classes for support, 2 classes for query
        
        num_shots = self.mifomo.num_shots_support
        
        num_shots_new_support = int(np.round(2 / 5 * num_shots))
        
        # We should have at least one sample for support or query sets.
        assert num_shots_new_support >= 1 and num_shots_new_support < num_shots
        
        support_indices_all_classes = torch.Tensor().long()
        query_indices_all_classes = torch.Tensor().long()
        
        for lbl in range(num_ways):
            indices_current_class_samples = torch.nonzero(episode_target.support_labels == lbl).squeeze()
            
            support_indices_current_class = indices_current_class_samples[:num_shots_new_support]
            query_indices_current_class = indices_current_class_samples[num_shots_new_support:]
            
            support_indices_all_classes = torch.cat([support_indices_all_classes, support_indices_current_class.cpu()])
            query_indices_all_classes = torch.cat([query_indices_all_classes, query_indices_current_class.cpu()])
            
        support_indices_all_classes = support_indices_all_classes.tolist()
        query_indices_all_classes = query_indices_all_classes.tolist()
        
        episode_target = sh.EpisodeContainer(
            support_samples=episode_target.support_samples[support_indices_all_classes].to(self.device),
            support_labels=episode_target.support_labels[support_indices_all_classes].to(self.device),
            support_labels_one_hot=episode_target.support_labels_one_hot[support_indices_all_classes].to(self.device),
            query_samples=episode_target.support_samples[query_indices_all_classes].to(self.device),
            query_labels=episode_target.support_labels[query_indices_all_classes].to(self.device),
            query_labels_one_hot=episode_target.support_labels_one_hot[query_indices_all_classes].to(self.device),
            num_ways=episode_target.num_ways
        )
        
        return episode_target
    
    @torch.no_grad()
    def _evaluate_episode(
        self,
        episode: sh.EpisodeContainer,
        # source_or_target=False,
    ):
        self.mifomo.eval()
        
        scores_without_label_smoothing, scores_with_label_smoothing = self.predict_query_labels(episode=episode)
        
        labels_query_predicted_without_label_smoothing = torch.argmax(scores_without_label_smoothing, dim=-1)
        # Overall Accuracy (OA)
        acc_without_label_smoothing = sh.calculate_accuracy(predictions=labels_query_predicted_without_label_smoothing, real_labels=episode.query_labels)
        
        # For Average Accuracy (AA)
        confusion_matrix = metrics.confusion_matrix(episode.query_labels.cpu().numpy(), labels_query_predicted_without_label_smoothing.cpu().numpy())
        A_without_label_smoothing = np.diag(confusion_matrix) / np.sum(confusion_matrix, 1, dtype=np.float64)
        
        # Kappa
        kappa_without_label_smoothing = metrics.cohen_kappa_score(episode.query_labels.cpu().numpy(), labels_query_predicted_without_label_smoothing.cpu().numpy())
        
        if self.mifomo.use_label_smoothing:
            labels_query_predicted_with_label_smoothing = torch.argmax(scores_with_label_smoothing, dim=-1)
        
            # Overall Accuracy (OA)
            acc_with_label_smoothing = sh.calculate_accuracy(predictions=labels_query_predicted_with_label_smoothing, real_labels=episode.query_labels)
        
            # For Average Accuracy (AA)
            confusion_matrix = metrics.confusion_matrix(episode.query_labels.cpu().numpy(), labels_query_predicted_with_label_smoothing.cpu().numpy())
            A_with_label_smoothing = np.diag(confusion_matrix) / np.sum(confusion_matrix, 1, dtype=np.float64)
            
            # Kappa
            kappa_with_label_smoothing = metrics.cohen_kappa_score(episode.query_labels.cpu().numpy(), labels_query_predicted_with_label_smoothing.cpu().numpy())
        else:
            acc_with_label_smoothing = 0.0
            A_with_label_smoothing = 0.0
            kappa_with_label_smoothing = 0.0
        
        return acc_without_label_smoothing, acc_with_label_smoothing, A_without_label_smoothing, A_with_label_smoothing, kappa_without_label_smoothing, kappa_with_label_smoothing
    
    def save_embedding_for_t_SNE(
        self,
    ):
        assert self.mifomo.num_episodes_tests == 1
        
        episode: sh.EpisodeContainer = self.intermediate_phase()
        
        episodeGenerator = sh.EpisodeGenerator(
            samples=self.mifomo.samples_target,
            samples_test=self.mifomo.samples_target_test,
            labels=self.mifomo.labels_target,
            labels_test=self.mifomo.labels_target_test,
            num_shots_support=0,   # The number of samples per class to show in a t-SNE plot.
            num_shots_query=-1,
            num_ways_limit=-1,
            device='cpu',
            dataset_name='t-SNE',
            keep_classes_with_insufficient_samples=True
        )
        
        episode = episodeGenerator.generate(calculate_one_hot_labels=False)
        
        self.mifomo.eval()
        samples = episode.query_samples
        labels = episode.query_labels
        # query_labels = episode.query_labels
        
        self.mifomo.eval()
        
        dataset = TensorDataset(samples, labels)
        dataloader = DataLoader(dataset, batch_size=80, drop_last=False)
        
        # embeddings_all = torch.Tensor()
        embeddings_all = torch.Tensor()
        labels_all = torch.Tensor().to(torch.long)
        
        with torch.no_grad():
            for x, labels in dataloader:
                embeddings = self.mifomo.forward(x, source_or_target=False).cpu()
            
                embeddings_all = torch.cat([embeddings_all, embeddings])
                labels_all = torch.cat([labels_all, labels])
            
        os.makedirs('t-SNE', exist_ok=True)
        torch.save({'embeddings': embeddings_all, 'labels': labels_all}, f't-SNE/embeddings_DS={self.mifomo.dataset_name_target}.pth')
            
    def draw_classification_map(
        self,
    ) -> None:
        
        assert self.mifomo.num_episodes_tests == 1
        
        episode: sh.EpisodeContainer = self.intermediate_phase()
        
        map_to_real_labels: T = episode.map_to_real_labels
        
        episode.move_to_device('cpu')
        
        gt_matrix = self.mifomo.gt_matrix_target
        gt_matrix_test = self.mifomo.gt_matrix_test_target
        height = gt_matrix.shape[0]
        width = gt_matrix.shape[1]
        
        gt_matrix_flattened = gt_matrix.flatten()
        
        classification_map_flattened = torch.zeros_like(gt_matrix_flattened)
        
        _, scores_with_label_smoothing = self.predict_query_labels(episode=episode)
        
        positions_labeled_samples = torch.nonzero(gt_matrix_flattened > 0, as_tuple=False).squeeze()
        indices_support_mapped = positions_labeled_samples[episode.support_set_indices_list]
        
        classification_map_flattened[indices_support_mapped] = episode.support_labels + 1                    # Background is zero.
        
        positions_labeled_samples_test = None
        if self.mifomo.dataset_name_target == 'Houston':
            gt_matrix_test_flattened = gt_matrix_test.flatten()
            positions_labeled_samples_test = torch.nonzero(gt_matrix_test_flattened > 0, as_tuple=False).squeeze()
            indices_query_mapped = positions_labeled_samples_test[episode.query_set_indices_list]
            
            # We merge all labels for the train and test set for the Ground truth classification map.
            gt_matrix_flattened[positions_labeled_samples_test] = gt_matrix_test_flattened[positions_labeled_samples_test]
        else:
            indices_query_mapped = positions_labeled_samples[episode.query_set_indices_list]
    
        classification_map_flattened[indices_query_mapped] = scores_with_label_smoothing.argmax(dim=-1) + 1  # Background is zero.
        
        classification_map_flattened = map_to_real_labels[classification_map_flattened]
        
        def assign_colors(classification_map_flattened: T, height: int, width: int):
            classification_map_RGB = torch.zeros(height * width, 3)
            # classification_map_RGB[classification_map == 0] = [0, 0, 0]
            classification_map_RGB[classification_map_flattened == 1] = torch.tensor([0, 0, 1], dtype=torch.float32)
            classification_map_RGB[classification_map_flattened == 2] = torch.tensor([0, 1, 0], dtype=torch.float32)
            classification_map_RGB[classification_map_flattened == 3] = torch.tensor([0, 1, 1], dtype=torch.float32)
            classification_map_RGB[classification_map_flattened == 4] = torch.tensor([1, 0, 0], dtype=torch.float32)
            classification_map_RGB[classification_map_flattened == 5] = torch.tensor([1, 0, 1], dtype=torch.float32)
            classification_map_RGB[classification_map_flattened == 6] = torch.tensor([1, 1, 0], dtype=torch.float32)
            classification_map_RGB[classification_map_flattened == 7] = torch.tensor([0.5, 0.5, 1], dtype=torch.float32)
            classification_map_RGB[classification_map_flattened == 8] = torch.tensor([0.65, 0.35, 1], dtype=torch.float32)
            classification_map_RGB[classification_map_flattened == 9] = torch.tensor([0.75, 0.5, 0.75], dtype=torch.float32)
            classification_map_RGB[classification_map_flattened == 10] = torch.tensor([0.75, 1, 0.5], dtype=torch.float32)
            classification_map_RGB[classification_map_flattened == 11] = torch.tensor([0.5, 1, 0.65], dtype=torch.float32)
            classification_map_RGB[classification_map_flattened == 12] = torch.tensor([0.65, 0.65, 0], dtype=torch.float32)
            classification_map_RGB[classification_map_flattened == 13] = torch.tensor([0.75, 1, 0.65], dtype=torch.float32)
            classification_map_RGB[classification_map_flattened == 14] = torch.tensor([0, 0, 0.5], dtype=torch.float32)
            classification_map_RGB[classification_map_flattened == 15] = torch.tensor([0, 1, 0.75], dtype=torch.float32)
            classification_map_RGB[classification_map_flattened == 16] = torch.tensor([0.5, 0.75, 1], dtype=torch.float32)
            classification_map_RGB = classification_map_RGB.view(height, width, 3)
            return classification_map_RGB
        
        classification_map_RGB = assign_colors(classification_map_flattened=classification_map_flattened, height=height, width=width)
        gt_matrix_RGB = assign_colors(classification_map_flattened=gt_matrix_flattened, height=height, width=width)
        
        path_classification_map = f"{self.mifomo.dir_classification_maps}/Classification_map,{self.mifomo.dataset_name_target},MIFOMO.png"
        path_gt_matrix = f"{self.mifomo.dir_classification_maps}/Classification_map,{self.mifomo.dataset_name_target},Ground_truth.png"
        
        utils.classification_map(
            map_np=classification_map_RGB.numpy(),
            dpi=24,
            savePath=path_classification_map
        )
        
        logging.info(f'Classification map is saved to "{path_classification_map}"')
        
        utils.classification_map(
            map_np=gt_matrix_RGB.numpy(),
            dpi=24,
            savePath=path_gt_matrix
        )
        
        logging.info(f'Ground truth map is saved to "{path_classification_map}"')
        
    def start(self):
        if self.mifomo.phase == nt.Phase.Source:
            self.source_phase()
        elif self.mifomo.phase == nt.Phase.Intermediate:
            self.intermediate_phase()
        elif self.mifomo.phase == nt.Phase.ClassificationMap:
            self.draw_classification_map()
        elif self.mifomo.phase == nt.Phase.t_SNE:
            self.save_embedding_for_t_SNE()
        else:
            raise NotImplementedError


def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--settings_file", type=str, default='configs/Indian_pines.json')
    args = parser.parse_args()
    
    stages = Stages(args.settings_file)
    
    stages.start()
    

if __name__ == '__main__':
    main()


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

# https://github.com/Naeem-Paeedeh/ADAPTER
# @article{paeedeh2024cross,
#   title={Cross-domain few-shot learning via adaptive transformer networks},
#   author={Paeedeh, Naeem and Pratama, Mahardhika and Ma’sum, Muhammad Anwar and Mayer, Wolfgang and Cao, Zehong and Kowlczyk, Ryszard},
#   journal={Knowledge-Based Systems},
#   volume={288},
#   pages={111458},
#   year={2024},
#   publisher={Elsevier}
# }

# https://github.com/Naeem-Paeedeh/CPLSR
# @article{paeedeh2025cross,
#   title={Cross-Domain Few-Shot Learning with Coalescent Projections and Latent Space Reservation},
#   author={Paeedeh, Naeem and Pratama, Mahardhika and Mayer, Wolfgang and Cao, Jimmy and Kowlczyk, Ryszard},
#   journal={arXiv preprint arXiv:2507.15243},
#   year={2025}
# }

# https://github.com/YuxiangZhang-BIT/IEEE_TNNLS_Gia-CFSL
# @article{zhang2022graph,
#   title={Graph information aggregation cross-domain few-shot learning for hyperspectral image classification},
#   author={Zhang, Yuxiang and Li, Wei and Zhang, Mengmeng and Wang, Shuai and Tao, Ran and Du, Qian},
#   journal={IEEE Transactions on Neural Networks and Learning Systems},
#   volume={35},
#   number={2},
#   pages={1912--1925},
#   year={2022},
#   publisher={IEEE}
# }

# https://github.com/jojolee6513/SCFormer
# @article{li2024scformer,
#   title={SCFormer: Spectral coordinate transformer for cross-domain few-shot hyperspectral image classification},
#   author={Li, Jiaojiao and Zhang, Zhiyuan and Song, Rui and Li, Yunsong and Du, Qian},
#   journal={IEEE transactions on image processing},
#   volume={33},
#   pages={840--855},
#   year={2024},
#   publisher={IEEE}
# }

# https://github.com/Li-ZK/CDFS-CASCL-2024/
# @article{li2024cross,
#   title={Cross-domain few-shot hyperspectral image classification with cross-modal alignment and supervised contrastive learning},
#   author={Li, Zhaokui and Zhang, Chenyang and Wang, Yan and Li, Wei and Du, Qian and Fang, Zhuoqun and Chen, Yushi},
#   journal={IEEE Transactions on Geoscience and Remote Sensing},
#   volume={62},
#   pages={1--19},
#   year={2024},
#   publisher={IEEE}
# }

# https://github.com/furqon3009/MDAN
# @article{Furqon2024MixupDA,
#   title={Mixup Domain Adaptations for Dynamic Remaining Useful Life Predictions},
#   author={Muhammad Tanzil Furqon and Mahardhika Pratama and Lin Liu and Habibullah Habibullah and Kutluyıl Doğançay},
#   journal={Knowl. Based Syst.},
#   year={2024},
#   volume={295},
#   pages={111783},
#   url={https://api.semanticscholar.org/CorpusID:269005807}
# }
