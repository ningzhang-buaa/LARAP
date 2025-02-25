import abc
import logging

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn as nn

import maple.torch.pytorch_util as ptu
from maple.policies.base import ExplorationPolicy
from maple.torch.core import torch_ify, elem_or_tuple_to_numpy
from maple.torch.distributions import (
    Delta, TanhNormal, MultivariateDiagonalNormal, GaussianMixture, GaussianMixtureFull,
)
from maple.torch.networks import Mlp, CNN
from maple.torch.networks.basic import MultiInputSequential
from maple.torch.networks.stochastic.distribution_generator import (
    DistributionGenerator
)


class TorchStochasticPolicy(
    DistributionGenerator,
    ExplorationPolicy, metaclass=abc.ABCMeta
):
    def get_action(self, epoch, obs_np, env_info, return_dist=False):
        #print('get_action')
        #print(env_info)
        # if env_info:
        #     print(env_info['grasped'])
        info = {}
        if return_dist:
            actions, dist = self.get_actions(epoch, obs_np[None], return_dist=return_dist)
            info['dist'] = dist
        else:
            actions = self.get_actions(epoch, obs_np[None], env_info, return_dist=return_dist)
        #if actions[0,0] < 0.5:
        #print(actions)
        return actions[0, :], info

    def get_actions(self, epoch, obs_np, env_info, return_dist=False):
        dist = self._get_dist_from_np(obs_np)
        # llm for lift task
        # if env_info:
        #     if env_info['grasped'] < 0.5:
        #         probs = torch.tensor([[0.25, 0.25, 0.5, 0., 0.]], device='cuda:0')
        #     else:
        #         probs = torch.tensor([[0.5, 0.5, 0.0, 0., 0.]], device='cuda:0')
        # else:
        #     probs = torch.tensor([[0.25, 0.25, 0.5, 0., 0.]], device='cuda:0')
        # end for lift task
        #print(dist.distr1.distr.probs)
        #probs = torch.tensor([[0.2, 0.2, 0.2, 0., 0.]], device='cuda:0')

        # llm for door task
        # if env_info:
        #     if env_info['d_reach'] > 0.05:
        #         probs = torch.tensor([[0.25, 0.25, 0.5, 0., 0.]], device='cuda:0')
        #     else:
        #         probs = torch.tensor([[0.5, 0.5, 0.0, 0., 0.]], device='cuda:0')
        # else:
        #     probs = torch.tensor([[0.25, 0.25, 0.5, 0., 0.]], device='cuda:0')
        #
        # if env_info:
        #     if env_info['success']:
        #         probs = torch.tensor([[0.5, 0.5, 0., 0., 0.]], device='cuda:0')
        # end for door task
        # if env_info:
        #     if env_info['reward_skills'] < 1.5:
        #         probs = torch.tensor([[0.25, 0.25, 0.5, 0., 0.]], device='cuda:0')
        #     else:
        #         probs = torch.tensor([[0.5, 0.5, 0.0, 0., 0.]], device='cuda:0')
        # else:
        #     probs = torch.tensor([[0.25, 0.25, 0.5, 0., 0.]], device='cuda:0')

        # if env_info:
        #     if env_info['reward_skills'] < 1.5:
        #         probs = torch.tensor([[0.3, 0.3, 0.3, 0., 0.]], device='cuda:0')
        #     else:
        #         probs = torch.tensor([[0.5, 0.5, 0.0, 0., 0.]], device='cuda:0')
        # else:
        #     probs = torch.tensor([[0.3, 0.3, 0.3, 0., 0.]], device='cuda:0')

        # llm for pnp task
        # probs = torch.tensor([[0.2, 0.2, 0.2, 0.2, 0.2]], device='cuda:0')
        # if env_info:
        #     if env_info['success'] < 0.1:
        #         if env_info['r_grasp'] < 0.5:
        #             probs = torch.tensor([[0.25, 0.25, 0.5, 0., 0.]], device='cuda:0')
        #         else:
        #             probs = torch.tensor([[0.5, 0.5, 0.0, 0., 0.]], device='cuda:0')
        #     else:
        #         probs = torch.tensor([[0.2, 0.2, 0.2, 0.2, 0.2]], device='cuda:0')
        # else:
        #     probs = torch.tensor([[0.3, 0.3, 0.3, 0., 0.]], device='cuda:0')
        #         # end for pnp task

        # llm for stack task
        # if env_info:
        #     if env_info['success'] < 0.1:
        #         if env_info['r_reach_grasp'] < 0.8:
        #             probs = torch.tensor([[0.25, 0.25, 0.5, 0., 0.]], device='cuda:0')
        #         else:
        #             probs = torch.tensor([[0.5, 0.5, 0.0, 0., 0.]], device='cuda:0')
        #     else:
        #         probs = torch.tensor([[0.2, 0.2, 0.2, 0.2, 0.2]], device='cuda:0')
        # else:
        #     probs = torch.tensor([[0.3, 0.3, 0.3, 0., 0.]], device='cuda:0')
        # end for stack task

        # llm for cleanup task
        # if env_info:
        #     if env_info['success'] < 0.1:
        #         if env_info['success_push'] < 0.1:
        #             probs = torch.tensor([[0.25, 0.25, 0., 0.5, 0.]], device='cuda:0')
        #         else:
        #             if env_info['r_grasp'] < 0.5:
        #                 probs = torch.tensor([[0.25, 0.25, 0.5, 0., 0.]], device='cuda:0')
        #             else:
        #                 if env_info['r_hover'] < 0.9:
        #                     probs = torch.tensor([[0.5, 0.5, 0.0, 0., 0.]], device='cuda:0')
        #                 else:
        #                     probs = torch.tensor([[0., 0., 0.0, 0., 0.5]], device='cuda:0')
        #     else:
        #         probs = torch.tensor([[0.2, 0.2, 0.2, 0.2, 0.2]], device='cuda:0')
        # else:
        #     probs = torch.tensor([[0.3, 0.3, 0.3, 0., 0.]], device='cuda:0')
        # end for cleanup task

        # llm for peg_ins task
        if env_info:
            if env_info['success'] < 0.1:
                if env_info['r_grasp'] < 0.8:
                    probs = torch.tensor([[0.25, 0.25, 0.5, 0., 0.]], device='cuda:0')
                else:
                    probs = torch.tensor([[0.5, 0.5, 0.0, 0., 0.]], device='cuda:0')
        else:
            probs = torch.tensor([[0.3, 0.3, 0.3, 0., 0.]], device='cuda:0')


        l = (500.0 -epoch) / 500.0

        if epoch == 10000:
            l=0.0
        if epoch == 5000:
            #print(epoch)
            l=0.0
        #if epoch >=0 and epoch <500:
            #print(epoch)
            #print(l)
            #print(probs)
        if (epoch != 6000 and epoch != 500):
            if (dist.__class__.__name__ == 'HierarchicalDistribution'):
                dist.distr1.renew_distr(probs * l)
        # if epoch > 0 and epoch < 500:
        #     print(dist.distr1.distr.probs)
        #print(dist.distr1.distr.probs)
        actions = dist.sample()
        #print(dist.distr1.sample())
        #print(actions)
        if return_dist:
            return elem_or_tuple_to_numpy(actions), dist
        else:
            return elem_or_tuple_to_numpy(actions)

    def _get_dist_from_np(self, *args, **kwargs):
        torch_args = tuple(torch_ify(x) for x in args)
        torch_kwargs = {k: torch_ify(v) for k, v in kwargs.items()}
        dist = self(*torch_args, **torch_kwargs)
        return dist


class PolicyFromDistributionGenerator(
    MultiInputSequential,
    TorchStochasticPolicy,
):
    """
    Usage:
    ```
    distribution_generator = FancyGenerativeModel()
    policy = PolicyFromBatchDistributionModule(distribution_generator)
    ```
    """
    pass


class MakeDeterministic(TorchStochasticPolicy):
    def __init__(
            self,
            action_distribution_generator: DistributionGenerator,
    ):
        super().__init__()
        self._action_distribution_generator = action_distribution_generator

    def forward(self, *args, **kwargs):
        dist = self._action_distribution_generator.forward(*args, **kwargs)
        return Delta(dist.mle_estimate())
