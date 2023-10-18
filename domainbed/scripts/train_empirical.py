# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import argparse
import collections
import json
import os
import random
import sys
import time
import uuid

import numpy as np
import PIL
import torch
import torchvision
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter


from domainbed import datasets
from domainbed import hparams_registry
from domainbed import algorithms
from domainbed.algorithms import PGD_Linf
from domainbed.lib import misc
from domainbed.lib.fast_data_loader import InfiniteDataLoader, FastDataLoader

from matplotlib import pyplot as plt
from torchvision import models
from autoattack import AutoAttack

import torch.nn as nn


# class PGD_Linf():

#     def __init__(self, model, epsilon=2/255, step_size=0.5/255, num_steps=10, random_start=True, target_mode=False, criterion='ce', bn_mode='eval', train=True):

#         self.model = model
#         self.epsilon = epsilon
#         self.step_size = step_size
#         self.num_steps = num_steps
#         self.random_start = random_start
#         self.target_mode = target_mode
#         self.bn_mode = bn_mode
#         self.train = train
#         self.criterion = criterion
#         self.criterion_ce = nn.CrossEntropyLoss()
#         self.criterion_kl = nn.KLDivLoss(reduction='sum')

#     def perturb(self, x_nat, targets=None):
#         if self.bn_mode == 'eval':
#             self.model.eval()

#         if self.random_start:
#             x_adv = x_nat.detach() + torch.empty_like(x_nat).uniform_(-self.epsilon,
#                                                                       self.epsilon).cuda().detach()
#             x_adv = torch.clamp(x_adv, min=0, max=1)
#         else:
#             x_adv = x_nat.clone().detach()

#         for _ in range(self.num_steps):
#             x_adv.requires_grad_()
#             outputs = self.model(x_adv)
#             # self.model.zero_grad()
#             if self.criterion == "ce":
#                 loss = self.criterion_ce(outputs, targets)
#                 loss.backward()
#                 grad = x_adv.grad
#             elif self.criterion == "kl":
#                 loss = self.criterion_kl(F.log_softmax(
#                     outputs, dim=1), F.softmax(self.model(x_nat), dim=1))
#                 grad = torch.autograd.grad(loss, [x_adv])[0]
#             elif self.criterion == "revkl":
#                 loss = self.criterion_kl(F.log_softmax(
#                     self.model(x_nat), dim=1), F.softmax(outputs, dim=1))
#                 grad = torch.autograd.grad(loss, [x_adv])[0]
#             elif self.criterion == "js":
#                 nat_probs = F.softmax(self.model(x_nat), dim=1)
#                 adv_probs = F.softmax(outputs, dim=1)
#                 mean_probs = (nat_probs + adv_probs)/2
#                 loss = (self.criterion_kl(mean_probs.log(), nat_probs) +
#                         self.criterion_kl(mean_probs.log(), adv_probs))/2
#                 grad = torch.autograd.grad(loss, [x_adv])[0]
#             if self.target_mode:
#                 x_adv = x_adv - self.step_size * grad.sign()
#             else:
#                 x_adv = x_adv + self.step_size * grad.sign()

#             d_adv = torch.clamp(x_adv - x_nat, min=-
#                                 self.epsilon, max=self.epsilon).detach()
#             x_adv = torch.clamp(x_nat + d_adv, min=0, max=1).detach()

#         if self.train:
#             self.model.train()

#         return x_adv, d_adv


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Domain generalization')
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--dataset', type=str, default="RotatedMNIST")
    parser.add_argument('--algorithm', type=str, default="ERM")
    parser.add_argument('--task', type=str, default="domain_generalization",
                        choices=["domain_generalization", "domain_adaptation"])
    parser.add_argument('--hparams', type=str,
                        help='JSON-serialized hparams dict')
    parser.add_argument('--hparams_seed', type=int, default=0,
                        help='Seed for random hparams (0 means "default hparams")')
    parser.add_argument('--trial_seed', type=int, default=0,
                        help='Trial number (used for seeding split_dataset and '
                        'random_hparams).')
    parser.add_argument('--seed', type=int, default=0,
                        help='Seed for everything else')
    parser.add_argument('--steps', type=int, default=None,
                        help='Number of steps. Default is dataset-dependent.')
    parser.add_argument('--checkpoint_freq', type=int, default=None,
                        help='Checkpoint every N steps. Default is dataset-dependent.')
    parser.add_argument('--test_envs', type=int, nargs='+', default=[0])
    parser.add_argument('--output_dir', type=str, default="train_output")
    parser.add_argument('--holdout_fraction', type=float, default=0.2)
    parser.add_argument('--uda_holdout_fraction', type=float, default=0,
                        help="For domain adaptation, % of test to use unlabeled for training.")
    parser.add_argument('--skip_model_save', action='store_true')
    parser.add_argument('--save_model_every_checkpoint', action='store_true')
    # TODO: add arguments for adversarial training
    parser.add_argument('--l2', type=float, default=0.0)
    parser.add_argument('--linf', type=float, default=0.0)
    parser.add_argument('--lam', type=float, default=0.5)

    args = parser.parse_args()

    # If we ever want to implement checkpointing, just persist these values
    # every once in a while, and then load them from disk here.
    start_step = 0
    algorithm_dict = None

    os.makedirs(args.output_dir, exist_ok=True)
    sys.stdout = misc.Tee(os.path.join(args.output_dir, 'out.txt'))
    sys.stderr = misc.Tee(os.path.join(args.output_dir, 'err.txt'))

    # print(f"domainbed/saved_imgs/{args.dataset}/Images_{str(args.test_envs)}")

    print("Environment:")
    print("\tPython: {}".format(sys.version.split(" ")[0]))
    print("\tPyTorch: {}".format(torch.__version__))
    print("\tTorchvision: {}".format(torchvision.__version__))
    print("\tCUDA: {}".format(torch.version.cuda))
    print("\tCUDNN: {}".format(torch.backends.cudnn.version()))
    print("\tNumPy: {}".format(np.__version__))
    print("\tPIL: {}".format(PIL.__version__))

    print('Args:')
    for k, v in sorted(vars(args).items()):
        print('\t{}: {}'.format(k, v))

    if args.hparams_seed == 0:
        hparams = hparams_registry.default_hparams(
            args.algorithm, args.dataset, args.l2, args.linf, args.lam)  # TODO
    else:
        hparams = hparams_registry.random_hparams(args.algorithm, args.dataset,
                                                  misc.seed_hash(args.hparams_seed, args.trial_seed))
    if args.hparams:
        hparams.update(json.loads(args.hparams))

    print('HParams:')
    for k, v in sorted(hparams.items()):
        print('\t{}: {}'.format(k, v))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    # Tensorboard

    if not os.path.exists("./runs"):
        os.makedirs("./runs")

    filename = "%s_%s_testenv_%s_lr_%.6f_weightd_%.6f_bs_%d" % (
        args.algorithm, args.dataset, args.test_envs, hparams["lr"], hparams["weight_decay"], hparams["batch_size"])
    writer = SummaryWriter("./runs/%s" % filename)

    if args.dataset in vars(datasets):
        dataset = vars(datasets)[args.dataset](args.data_dir,
                                               args.test_envs, hparams)
    else:
        raise NotImplementedError

    # Split each env into an 'in-split' and an 'out-split'. We'll train on
    # each in-split except the test envs, and evaluate on all splits.

    # To allow unsupervised domain adaptation experiments, we split each test
    # env into 'in-split', 'uda-split' and 'out-split'. The 'in-split' is used
    # by collect_results.py to compute classification accuracies.  The
    # 'out-split' is used by the Oracle model selectino method. The unlabeled
    # samples in 'uda-split' are passed to the algorithm at training time if
    # args.task == "domain_adaptation". If we are interested in comparing
    # domain generalization and domain adaptation results, then domain
    # generalization algorithms should create the same 'uda-splits', which will
    # be discared at training.
    in_splits = []
    out_splits = []
    uda_splits = []
    for env_i, env in enumerate(dataset):
        uda = []

        out, in_ = misc.split_dataset(env,
                                      int(len(env)*args.holdout_fraction),
                                      misc.seed_hash(args.trial_seed, env_i))

        if env_i in args.test_envs:
            uda, in_ = misc.split_dataset(in_,
                                          int(len(in_)*args.uda_holdout_fraction),
                                          misc.seed_hash(args.trial_seed, env_i))

        if hparams['class_balanced']:
            in_weights = misc.make_weights_for_balanced_classes(in_)
            out_weights = misc.make_weights_for_balanced_classes(out)
            if uda is not None:
                uda_weights = misc.make_weights_for_balanced_classes(uda)
        else:
            in_weights, out_weights, uda_weights = None, None, None
        in_splits.append((in_, in_weights))
        out_splits.append((out, out_weights))
        if len(uda):
            uda_splits.append((uda, uda_weights))

    if args.task == "domain_adaptation" and len(uda_splits) == 0:
        raise ValueError("Not enough unlabeled samples for domain adaptation.")

    train_loaders = [InfiniteDataLoader(
        dataset=env,
        weights=env_weights,
        batch_size=hparams['batch_size'],
        num_workers=dataset.N_WORKERS)
        for i, (env, env_weights) in enumerate(in_splits)
        if i not in args.test_envs]

    uda_loaders = [InfiniteDataLoader(
        dataset=env,
        weights=env_weights,
        batch_size=hparams['batch_size'],
        num_workers=dataset.N_WORKERS)
        for i, (env, env_weights) in enumerate(uda_splits)
        if i in args.test_envs]

    eval_loaders = [FastDataLoader(
        dataset=env,
        batch_size=64,
        num_workers=dataset.N_WORKERS)
        for env, _ in (in_splits + out_splits + uda_splits)]

    ### TODO: Double check code here ###

    eval_out_src = [FastDataLoader(
        dataset=env,
        batch_size=hparams['batch_size'],
        num_workers=dataset.N_WORKERS)
        for i, (env, env_weights) in enumerate(out_splits)
        if i not in args.test_envs]

    eval_tgt = [FastDataLoader(
        dataset=env,
        batch_size=hparams['batch_size'],
        num_workers=dataset.N_WORKERS)
        for i, (env, env_weights) in enumerate(in_splits)
        if i in args.test_envs] + [FastDataLoader(
            dataset=env,
            batch_size=hparams['batch_size'],
            num_workers=dataset.N_WORKERS)
        for i, (env, env_weights) in enumerate(out_splits)
        if i in args.test_envs]

    eval_weights = [None for _, weights in (
        in_splits + out_splits + uda_splits)]
    eval_loader_names = ['env{}_in'.format(i)
                         for i in range(len(in_splits))]
    eval_loader_names += ['env{}_out'.format(i)
                          for i in range(len(out_splits))]
    eval_loader_names += ['env{}_uda'.format(i)
                          for i in range(len(uda_splits))]

    algorithm_class = algorithms.get_algorithm_class(args.algorithm)
    algorithm = algorithm_class(dataset.input_shape, dataset.num_classes,
                                len(dataset) - len(args.test_envs), hparams)

    if algorithm_dict is not None:
        algorithm.load_state_dict(algorithm_dict)

    algorithm.to(device)

    train_minibatches_iterator = zip(*train_loaders)
    uda_minibatches_iterator = zip(*uda_loaders)
    checkpoint_vals = collections.defaultdict(lambda: [])

    steps_per_epoch = min([len(env)/hparams['batch_size']
                          for env, _ in in_splits])

    n_steps = args.steps or dataset.N_STEPS
    checkpoint_freq = args.checkpoint_freq or dataset.CHECKPOINT_FREQ

    def save_checkpoint(filename):
        if args.skip_model_save:
            return
        save_dict = {
            "args": vars(args),
            "model_input_shape": dataset.input_shape,
            "model_num_classes": dataset.num_classes,
            "model_num_domains": len(dataset) - len(args.test_envs),
            "model_hparams": hparams,
            "model_dict": algorithm.state_dict()
        }
        torch.save(save_dict, os.path.join(args.output_dir, filename))

    with open(os.path.join(args.output_dir, 'inprogress'), 'w') as f:
        f.write('inprogress')

    last_results_keys = None

    # Initialize variables to track the best accuracy and corresponding information
    best_accuracy_in = 0
    best_src_acc_out = 0
    best_step_in = 0
    best_name_in = "None"

    best_accuracy_out = 0
    best_step_out = 0
    best_name_out = "None"

    best_acc = 0
    tgt_final_acc = 0

    # Loop through the specified number of steps
    for step in range(start_step, n_steps):

        algorithm.train()

        step_start_time = time.time()
        minibatches_device = [(x.to(device), y.to(device))
                              for x, y in next(train_minibatches_iterator)]
        if args.task == "domain_adaptation":
            uda_device = [x.to(device)
                          for x, _ in next(uda_minibatches_iterator)]
        else:
            uda_device = None

        step_vals = algorithm.update(minibatches_device, uda_device)
        checkpoint_vals['step_time'].append(time.time() - step_start_time)

        for key, val in step_vals.items():
            checkpoint_vals[key].append(val)

        # Checkpointing: Save and evaluate the model at specified frequencies
        if (step % checkpoint_freq == 0) or (step == n_steps - 1):
            results = {
                'step': step,
                'epoch': step / steps_per_epoch,
            }

            for key, val in checkpoint_vals.items():
                results[key] = np.mean(val)

            # Evaluate the model on different evaluation loaders
            evals = zip(eval_loader_names, eval_loaders, eval_weights)
            src_acc = 0

            for name, loader, weights in evals:
                acc = misc.accuracy(algorithm, loader, weights, device)
                results[name+'_acc'] = acc

            algorithm.eval()

            for loader in eval_tgt:
                correct = 0
                total = 0
                for batch_idx, (images, labels) in enumerate(loader):
                    with torch.no_grad():
                        images = images.cuda()
                        labels = labels.cuda()
                        outputs = algorithm.predict(images)
                        _, predicted = torch.max(outputs.data, 1)
                        correct += (predicted == labels).sum().item()
                        total += images.shape[0]

            tgt_acc = correct/total * 100

            src_acc = 0
            for loader in eval_out_src:
                correct = 0
                total = 0
                for batch_idx, (images, labels) in enumerate(loader):
                    with torch.no_grad():
                        images = images.cuda()
                        labels = labels.cuda()
                        outputs = algorithm.predict(images)
                        _, predicted = torch.max(outputs.data, 1)
                        correct += (predicted == labels).sum().item()
                        total += images.shape[0]
                src_acc += correct/total * 100

            src_acc = src_acc/len(eval_out_src)

            # Select the best target accuracy based on the best source validation checkpoint
            if src_acc > best_acc:
                best_acc = src_acc
                tgt_final_acc = tgt_acc
                try:
                    model = algorithm.net
                except:
                    model = algorithm.network

                torch.save(model.state_dict(), os.path.join(
                    args.output_dir, 'best_checkpoint_%s.pt' % (args.test_envs)))

            algorithm.train()

            # Print accuracies to file
            with open(os.path.join(args.output_dir, 'src_val_accs.txt'), "a") as text_file:
                print(f'{step} {src_acc}', file=text_file)

            with open(os.path.join(args.output_dir, 'tgt_accs.txt'), "a") as text_file:
                print(f'{step} {tgt_acc}', file=text_file)

            results['mem_gb'] = torch.cuda.max_memory_allocated() / \
                (1024.*1024.*1024.)

            results_keys = sorted(results.keys())
            if results_keys != last_results_keys:
                misc.print_row(results_keys, colwidth=12)
                last_results_keys = results_keys
            misc.print_row([results[key] for key in results_keys],
                           colwidth=12)

            results.update({
                'hparams': hparams,
                'args': vars(args)
            })

            epochs_path = os.path.join(args.output_dir, 'results.jsonl')
            with open(epochs_path, 'a') as f:
                f.write(json.dumps(results, sort_keys=True) + "\n")

            algorithm_dict = algorithm.state_dict()
            start_step = step + 1
            checkpoint_vals = collections.defaultdict(lambda: [])

            if args.save_model_every_checkpoint:
                save_checkpoint(f'model_step{step}.pkl')

    # Save the final model
    save_checkpoint('model.pkl')

    # Write the best accuracies to TensorBoard
    writer.add_scalar('best_accuracy/%s' % best_name_in,
                      best_accuracy_in, best_step_in)
    writer.add_scalar('best_accuracy/%s' % best_name_out,
                      best_accuracy_out, best_step_out)

    with open(os.path.join(args.output_dir, 'done'), 'w') as f:
        f.write('done')

    try:
        model = algorithm.net
    except:
        model = algorithm.network

    # Best model analysis
    model.load_state_dict(torch.load(os.path.join(
        args.output_dir, 'best_checkpoint_%s.pt' % (args.test_envs))))
    model.eval()

    # AutoAttack evaluation
    src_robust_acc = 0

    # Evaluate on source domains
    for loader in eval_out_src:
        correct = 0
        total = 0
        num_samples = 0

        # Iterate through the dataset in chunks
        for batch_idx, (inputs, targets) in enumerate(loader):

            inputs = inputs.cuda()
            targets = targets.cuda()

            # Perform AutoAttack with Linf norm and epsilon=2/255
            adversary = AutoAttack(
                model, norm='Linf', eps=2/255, version='standard')
            adversary.attacks_to_run = ['apgd-ce']
            x_adv = adversary.run_standard_evaluation(
                inputs, targets, bs=inputs.shape[0])

            # Compute robust accuracy
            correct += torch.logical_and((model(x_adv).argmax(1) ==
                                         targets), (model(inputs).argmax(1) == targets)).sum()
            total += x_adv.shape[0]

            num_samples += inputs.shape[0]
            print(num_samples)
            if num_samples >= 256:
                break

        src_robust_acc += correct/total*100
    src_robust_acc = src_robust_acc/len(eval_out_src)

    # Evaluate on target domain
    for loader in eval_tgt:
        correct = 0
        total = 0
        num_samples = 0

        # Iterate through the dataset in chunks
        for batch_idx, (inputs, targets) in enumerate(loader):

            inputs = inputs.cuda()
            targets = targets.cuda()

            # Perform AutoAttack with Linf norm and epsilon=2/255
            adversary = AutoAttack(
                model, norm='Linf', eps=2/255, version='standard')
            adversary.attacks_to_run = ['apgd-ce']
            x_adv = adversary.run_standard_evaluation(
                inputs, targets, bs=inputs.shape[0])

            # Compute robust accuracy
            correct += torch.logical_and((model(x_adv).argmax(1) ==
                                         targets), (model(inputs).argmax(1) == targets)).sum()
            total += x_adv.shape[0]

            num_samples += inputs.shape[0]
            print(num_samples)
            if num_samples >= 256:
                break
    tgt_robust_acc = correct/total*100

    # Save the Autoattack Robustness and Clean Accuracy results to file
    with open(os.path.join(args.output_dir, 'src_robustness_AA.txt'), "a") as text_file:
        print(f'{src_robust_acc}', file=text_file)

    with open(os.path.join(args.output_dir, 'tgt_robustness_AA.txt'), "a") as text_file:
        print(f'{tgt_robust_acc}', file=text_file)

    with open(os.path.join(args.output_dir, 'src_acc_clean.txt'), "a") as text_file:
        print(f'{best_acc}', file=text_file)

    with open(os.path.join(args.output_dir, 'tgt_acc_clean.txt'), "a") as text_file:
        print(f'{tgt_final_acc}', file=text_file)

    # PGD evaluation
    src_robust_acc = 0

    # Evaluate on source domains
    for loader in eval_out_src:
        correct = 0
        total = 0
        num_samples = 0

        for batch_idx, (inputs, targets) in enumerate(loader):

            inputs = inputs.cuda()
            targets = targets.cuda()

            # Perform PGD attack with Linf norm and epsilon=2/255
            x_adv = PGD_Linf(model, epsilon=2/255, step_size=0.05/255, num_steps=20, random_start=True,
                             target_mode=False, criterion='ce', bn_mode='eval', train=False).perturb(inputs, targets)[0]

            # Compute robust accuracy
            correct += torch.logical_and((model(x_adv).argmax(1) ==
                                         targets), (model(inputs).argmax(1) == targets)).sum()
            total += x_adv.shape[0]

            num_samples += inputs.shape[0]
            print(num_samples)
            if num_samples >= 256:
                break

        src_robust_acc += correct/total*100
    src_robust_acc = src_robust_acc/len(eval_out_src)

    # Evaluate on target domain
    for loader in eval_tgt:
        correct = 0
        total = 0
        num_samples = 0
        for batch_idx, (inputs, targets) in enumerate(loader):

            inputs = inputs.cuda()
            targets = targets.cuda()

            # Perform PGD attack with Linf norm and epsilon=2/255
            x_adv = PGD_Linf(model, epsilon=2/255, step_size=0.05/255, num_steps=20, random_start=True,
                             target_mode=False, criterion='ce', bn_mode='eval', train=False).perturb(inputs, targets)[0]

            # Compute robust accuracy
            correct += torch.logical_and((model(x_adv).argmax(1) ==
                                         targets), (model(inputs).argmax(1) == targets)).sum()
            total += x_adv.shape[0]

            num_samples += inputs.shape[0]
            print(num_samples)
            if num_samples >= 256:
                break
    tgt_robust_acc = correct/total*100

    # Save the results to file
    with open(os.path.join(args.output_dir, 'src_robustness_PGD.txt'), "a") as text_file:
        print(f'{src_robust_acc}', file=text_file)

    with open(os.path.join(args.output_dir, 'tgt_robustness_PGD.txt'), "a") as text_file:
        print(f'{tgt_robust_acc}', file=text_file)
