# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import copy
import os
from ast import Raise
from collections import OrderedDict

import eagerpy as ep
import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F

try:
    from backpack import backpack, extend
    from backpack.extensions import BatchGrad
except:
    backpack = None

import numpy as np

from domainbed import networks
from domainbed.lib.misc import (
    MovingAverage,
    ParamDict,
    l2_between_dicts,
    random_pairs_of_minibatches,
)
from domainbed.trades import *

eps = 2 / 255
step = eps / 4
num_steps = 10
beta = 3.0

ALGORITHMS = [
    'ERM', 'PGDLinf', 'TradesLinf', 'PGDL2', 'TradesL2'
]


def get_algorithm_class(algorithm_name):
    """Return the algorithm class with the given name."""
    if algorithm_name not in globals():
        raise NotImplementedError(
            "Algorithm not found: {}".format(algorithm_name))
    return globals()[algorithm_name]


class Algorithm(torch.nn.Module):
    """
    A subclass of Algorithm implements a domain generalization algorithm.
    Subclasses should implement the following:
    - update()
    - predict()
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(Algorithm, self).__init__()
        self.hparams = hparams

    def update(self, minibatches, unlabeled=None):
        """
        Perform one update step, given a list of (x, y) tuples for all
        environments.

        Admits an optional list of unlabeled minibatches from the test domains,
        when task is domain_adaptation.
        """
        raise NotImplementedError

    def predict(self, x):
        raise NotImplementedError


class ERM(Algorithm):
    """
    Empirical Risk Minimization (ERM)

    Args:
        input_shape (tuple): Shape of the input data.
        num_classes (int): Number of output classes.
        num_domains (int): Number of domains in the dataset.
        hparams (dict): Additional hyperparameters for the algorithm.

    Attributes:
        featurizer (networks.Featurizer): Featurizer network for extracting features from input data.
        classifier (networks.Classifier): Classifier network for predicting class labels.
        network (torch.nn.Sequential): Sequential model combining the featurizer and classifier.
        optimizer (torch.optim.Adam): Optimizer used for training the network.
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(ERM, self).__init__(input_shape, num_classes, num_domains,
                                  hparams)
        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = networks.Classifier(
            self.featurizer.n_outputs, num_classes,
            self.hparams['nonlinear_classifier'])

        self.network = nn.Sequential(self.featurizer, self.classifier)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay'])

    def update(self, minibatches, unlabeled=None):
        """
        Perform one update step using the provided minibatches.

        Args:
            minibatches (list): List of (x, y) tuples for all environments.
            unlabeled (list, optional): List of unlabeled minibatches from the test domains.

        Returns:
            dict: Dictionary containing the loss value.

        """
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        loss = F.cross_entropy(self.predict(all_x), all_y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {'loss': loss.item()}

    def predict(self, x):
        """
        Make predictions for the given input data.

        Args:
            x (torch.Tensor): Input data tensor.

        Returns:
            torch.Tensor: Predicted output tensor.

        """
        return self.network(x)


class PGDLinf(ERM):
    """
    Reimplementation of PGD-Linf Attack

    This class represents a reimplementation of the PGD-Linf attack as a subclass of ERM.
    It inherits the common functionality from ERM and overrides the update method for PGD-Linf attack.

    Args:
        input_shape (tuple): Shape of the input data.
        num_classes (int): Number of output classes.
        num_domains (int): Number of domains in the dataset.
        hparams (dict): Additional hyperparameters for the algorithm.

    Attributes:
        featurizer (networks.Featurizer): Featurizer network for extracting features from input data.
        classifier (networks.Classifier): Classifier network for predicting class labels.
        network (torch.nn.Sequential): Sequential model combining the featurizer and classifier.
        optimizer (torch.optim.Adam): Optimizer used for training the network.
    """
    
    def update(self, minibatches, unlabeled=None):
        """
        Perform one update step using the provided minibatches.
        """
        # Concatenate all input and target tensors from minibatches
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        
        # Perform PGD-Linf attack to obtain perturbed examples
        all_x_pert = PGD_Linf(
            self.network,
            epsilon=eps,
            step_size=step,
            num_steps=num_steps,
            random_start=True,
            target_mode=False,
            criterion='ce',
            bn_mode='eval',
            train=True).perturb(all_x, all_y)[0]

        # Compute the loss as a combination of the original and perturbed predictions
        loss = 0.5 * F.cross_entropy(self.predict(all_x), all_y) + \
            0.5 * F.cross_entropy(self.predict(all_x_pert), all_y)

        # Perform backpropagation and optimization
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Return the loss value as a dictionary
        return {'loss': loss.item()}


class TradesLinf(ERM):
    """
    Uses TRADES (TRadeoff-inspired Adversarial DEfense via Surrogate-loss minimization)

    This class represents an implementation of the TRADES defense as a subclass of ERM.
    It inherits the common functionality from ERM and overrides the update method for TRADES.

    Args:
        input_shape (tuple): Shape of the input data.
        num_classes (int): Number of output classes.
        num_domains (int): Number of domains in the dataset.
        hparams (dict): Additional hyperparameters for the algorithm.
    """

    def update(self, minibatches, unlabeled=None):
        """
        Perform one update step using the provided minibatches and TRADES loss.
        """        
        # Perform TRADES loss calculation on the model's featurizer and classifier
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        
        loss = trades_loss(
            self.network,
            all_x,
            all_y,
            self.optimizer,
            step_size=step,
            epsilon=eps,
            perturb_steps=num_steps,
            beta=beta,
            distance='l_inf'
        )
        
        # Perform backpropagation and optimization
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Return the loss value as a dictionary
        return {'loss': loss.item()}


class PGDL2(ERM):
    """
    Reimplementation of PGD-L2 Attack

    This class represents a reimplementation of the PGD-L2 attack as a subclass of ERM.
    It inherits the common functionality from ERM and overrides the update method for PGD-L2 attack.

    Args:
        input_shape (tuple): Shape of the input data.
        num_classes (int): Number of output classes.
        num_domains (int): Number of domains in the dataset.
        hparams (dict): Additional hyperparameters for the algorithm.
    """

    def update(self, minibatches, unlabeled=None):
        """
        Perform one update step using the provided minibatches.
        """
        # Concatenate all input and target tensors from minibatches
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        
        # Perform PGD-L2 attack to obtain perturbed examples
        eps = 2
        step = eps / 5

        all_x_pert = PGD_L2(self.network,
                            epsilon=eps,
                            step_size=step,
                            num_steps=20,
                            random_start=True,
                            target_mode=False,
                            criterion='ce',
                            bn_mode='eval',
                            train=True).perturb(all_x, all_y)[0]
        
        # Compute the loss as a combination of the original and perturbed predictions
        loss = 0.5 * F.cross_entropy(self.predict(all_x), all_y) + \
            0.5 * F.cross_entropy(self.predict(all_x_pert), all_y)
            
        # Perform backpropagation and optimization
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Return the loss value as a dictionary
        return {'loss': loss.item()}

class TradesL2(ERM):
    """
    Uses TRADES (TRadeoff-inspired Adversarial DEfense via Surrogate-loss minimization)

    This class represents an implementation of the TRADES defense as a subclass of ERM.
    It inherits the common functionality from ERM and overrides the update method for TRADES.

    Args:
        input_shape (tuple): Shape of the input data.
        num_classes (int): Number of output classes.
        num_domains (int): Number of domains in the dataset.
        hparams (dict): Additional hyperparameters for the algorithm.
    """

    def update(self, minibatches, unlabeled=None):
        """
        Perform one update step using the provided minibatches and TRADES loss.
        """        
        # Perform TRADES loss calculation on the model's featurizer and classifier
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])

        eps = 2
        step = eps / 5

        loss = trades_loss(self.network,
                           all_x,
                           all_y,
                           self.optimizer,
                           step_size=step,
                           epsilon=eps,
                           perturb_steps=5,
                           beta=3,
                           distance='l_2')

        # Perform backpropagation and optimization
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Return the loss value as a dictionary
        return {'loss': loss.item()}

class PGD_Linf():
    """
    PGD-Linf Attack

    This class represents the PGD-Linf attack. It generates adversarial examples
    using the L-infinity norm and performs perturbations on input examples.

    Args:
        model (torch.nn.Module): The model to attack.
        epsilon (float, optional): The maximum perturbation limit (default: 8/255).
        step_size (float, optional): The step size for each iteration (default: 2/255).
        num_steps (int, optional): The number of steps for the attack (default: 10).
        random_start (bool, optional): Whether to start from a random perturbation (default: True).
        target_mode (bool, optional): Whether to perform targeted attack (default: False).
        criterion (str, optional): The loss criterion to optimize ('ce', 'kl', 'revkl', 'js') (default: 'ce').
        bn_mode (str, optional): Batch normalization mode ('eval' or 'train') (default: 'eval').
        train (bool, optional): Whether the model is in training mode (default: True).

    Methods:
        perturb(x_nat, targets=None):
            Generate adversarial examples by perturbing the input examples.

    """

    def __init__(self,
                 model,
                 epsilon=8 / 255,
                 step_size=2 / 255,
                 num_steps=10,
                 random_start=True,
                 target_mode=False,
                 criterion='ce',
                 bn_mode='eval',
                 train=True):
        """
        Initialize the PGD_Linf attack with the specified parameters.
        """
        self.model = model
        self.epsilon = epsilon
        self.step_size = step_size
        self.num_steps = num_steps
        self.random_start = random_start
        self.target_mode = target_mode
        self.bn_mode = bn_mode
        self.train = train
        self.criterion = criterion
        self.criterion_ce = nn.CrossEntropyLoss()
        self.criterion_kl = nn.KLDivLoss(reduction='sum')

    def perturb(self, x_nat, targets=None):
        """
        Generate adversarial examples by perturbing the input examples.

        Args:
            x_nat (torch.Tensor): The original input examples.
            targets (torch.Tensor, optional): The target labels for targeted attack (default: None).

        Returns:
            torch.Tensor: Perturbed adversarial examples.
            torch.Tensor: Difference between the perturbed examples and the original examples.

        """
        if self.bn_mode == 'eval':
            self.model.eval()

        if self.random_start:
            x_adv = x_nat.detach() + torch.empty_like(x_nat).uniform_(
                -self.epsilon, self.epsilon).cuda().detach()
            x_adv = torch.clamp(x_adv, min=0, max=1)
        else:
            x_adv = x_nat.clone().detach()

        for _ in range(self.num_steps):
            x_adv.requires_grad_()
            outputs = self.model(x_adv)

            if self.criterion == "ce":
                loss = self.criterion_ce(outputs, targets)
                loss.backward()
                grad = x_adv.grad
            elif self.criterion == "kl":
                loss = self.criterion_kl(F.log_softmax(outputs, dim=1),
                                         F.softmax(self.model(x_nat), dim=1))
                grad = torch.autograd.grad(loss, [x_adv])[0]
            elif self.criterion == "revkl":
                loss = self.criterion_kl(
                    F.log_softmax(self.model(x_nat), dim=1),
                    F.softmax(outputs, dim=1))
                grad = torch.autograd.grad(loss, [x_adv])[0]
            elif self.criterion == "js":
                nat_probs = F.softmax(self.model(x_nat), dim=1)
                adv_probs = F.softmax(outputs, dim=1)
                mean_probs = (nat_probs + adv_probs) / 2
                loss = (self.criterion_kl(mean_probs.log(), nat_probs) +
                        self.criterion_kl(mean_probs.log(), adv_probs)) / 2
                grad = torch.autograd.grad(loss, [x_adv])[0]
            if self.target_mode:
                x_adv = x_adv - self.step_size * grad.sign()
            else:
                x_adv = x_adv + self.step_size * grad.sign()

            d_adv = torch.clamp(x_adv - x_nat,
                                min=-self.epsilon,
                                max=self.epsilon).detach()
            x_adv = torch.clamp(x_nat + d_adv, min=0, max=1).detach()

        if self.train:
            self.model.train()

        return x_adv, d_adv


class PGD_L2():
    """
    PGD-L2 Attack

    This class represents the PGD-L2 attack. It generates adversarial examples
    using the L2 norm and performs perturbations on input examples.

    Args:
        model (torch.nn.Module): The model to attack.
        epsilon (float, optional): The maximum perturbation limit (default: 20/255).
        step_size (float, optional): The step size for each iteration (default: 4/255).
        num_steps (int, optional): The number of steps for the attack (default: 10).
        random_start (bool, optional): Whether to start from a random perturbation (default: True).
        target_mode (bool, optional): Whether to perform targeted attack (default: False).
        criterion (str, optional): The loss criterion to optimize ('ce', 'kl', 'revkl') (default: 'ce').
        bn_mode (str, optional): Batch normalization mode ('eval' or 'train') (default: 'eval').
        train (bool, optional): Whether the model is in training mode (default: True).

    Methods:
        perturb(x_nat, targets):
            Generate adversarial examples by perturbing the input examples.

    """

    def __init__(self,
                 model,
                 epsilon=20 / 255,
                 step_size=4 / 255,
                 num_steps=10,
                 random_start=True,
                 target_mode=False,
                 criterion='ce',
                 bn_mode='eval',
                 train=True):
        """
        Initialize the PGD_L2 attack with the specified parameters.
        """
        self.model = model
        self.epsilon = epsilon
        self.step_size = step_size
        self.num_steps = num_steps
        self.random_start = random_start
        self.target_mode = target_mode
        self.bn_mode = bn_mode
        self.train = train
        self.criterion = criterion
        self.criterion_ce = nn.CrossEntropyLoss()
        self.criterion_kl = nn.KLDivLoss(reduction='sum')

    def perturb(self, x_nat, targets):
        """
        Generate adversarial examples by perturbing the input examples.

        Args:
            x_nat (torch.Tensor): The original input examples.
            targets (torch.Tensor): The target labels for targeted attack.

        Returns:
            torch.Tensor: Perturbed adversarial examples.
            torch.Tensor: Difference between the perturbed examples and the original examples.

        """
        if self.bn_mode == 'eval':
            self.model.eval()

        if self.random_start:
            x_adv = x_nat.detach() + torch.empty_like(x_nat).uniform_(
                -self.epsilon, self.epsilon).cuda().detach()
            x_adv = torch.clamp(x_adv, min=0, max=1)
        else:
            x_adv = x_nat.clone().detach()

        for _ in range(self.num_steps):

            x_adv.requires_grad_()
            outputs = self.model(x_adv)
            self.model.zero_grad()
            if self.criterion == "ce":
                loss = self.criterion_ce(outputs, targets)
                loss.backward()
                grad = x_adv.grad
            elif self.criterion == "kl":
                loss = self.criterion_kl(F.log_softmax(outputs, dim=1),
                                         F.softmax(self.model(x_nat), dim=1))
                grad = torch.autograd.grad(loss, [x_adv])[0]
            elif self.criterion == "revkl":
                loss = self.criterion_kl(
                    F.log_softmax(self.model(x_nat), dim=1),
                    F.softmax(outputs, dim=1))
                grad = torch.autograd.grad(loss, [x_adv])[0]

            grad_norm = grad.abs().pow(2).view(x_nat.shape[0],
                                               -1).sum(1).pow(1. / 2)
            grad_norm = grad_norm.view(x_nat.shape[0], 1, 1,
                                       1).expand_as(x_nat)
            d_adv = grad / grad_norm

            if self.target_mode:
                x_adv = x_adv - self.step_size * d_adv
            else:
                x_adv = x_adv + self.step_size * d_adv

            d_adv = (x_adv - x_nat).view(x_nat.shape[0], -1).detach()
            d_adv = d_adv.view(x_nat.shape)
            d_adv = torch.clamp(d_adv, min=-self.epsilon, max=self.epsilon)

            x_adv = torch.clamp(x_nat + d_adv, min=0, max=1).detach()

        if self.train:
            self.model.train()

        return x_adv, d_adv
