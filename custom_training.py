"""
Custom code for Pl@ntNet-300K but using mobilevit v2 from HuggingFace
"""
import os
import time
from collections import defaultdict

import torch
from torch.cuda.amp import GradScaler
from torch.nn import functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from constants import LIST_K
from utils import count_correct_topk, count_correct_avgk, update_correct_per_class, \
    update_correct_per_class_topk, update_correct_per_class_avgk, set_seed, LabelSmoothingCrossEntropy, lr_lambda, save


class TreeClassificationModel:
    def __init__(self, model):
        self.model = model
        self.train_history = []
        self.train_losses = []
        self.train_acc = []
        self.val_losses = []
        self.val_acc = []
        self.val_topk_acc = []
        self.val_avgk_acc = []
        self.val_class_acc = []
        self.test_acc = []
        self.test_topk_acc = []
        self.test_avgk_acc = []
        self.train_topk_acc = []

    def train_epoch(
            self,
            train_loader,
            optimizer,
            scheduler,
            criteria,
            use_gpu,
            fp16,
            scaler
    ):
        self.model.train()
        total_loss, total_correct = 0, 0
        total_samples = 0
        n_correct_topk_train = defaultdict(int)
        topk_acc_epoch_train = {}

        for batch_idx, (batch_x_train, batch_y_train) in enumerate(tqdm(train_loader, desc='train', position=0)):
            if use_gpu:
                batch_x_train, batch_y_train = batch_x_train.cuda(), batch_y_train.cuda()

            optimizer.zero_grad()
            if fp16:
                with torch.amp.autocast(device_type='cuda'):
                    outputs = self.model(batch_x_train).logits
                    loss = criteria(outputs, batch_y_train)
            else:
                outputs = self.model(batch_x_train).logits
                loss = criteria(outputs, batch_y_train)

            # The gradient scaler will become no op if it detects the system is using fp32 instead
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if scheduler:
                scheduler.step()
            train_sizes = batch_x_train.size(0)
            total_loss += loss.item()
            total_correct += (outputs.argmax(dim=-1) == batch_y_train).sum().item()
            total_samples += train_sizes

            # Get top_k accuracy
            for k in LIST_K:
                n_correct_topk_train[k] += count_correct_topk(scores=outputs, labels=batch_y_train, k=k).item()
                topk_acc_epoch_train[k] = n_correct_topk_train[k] / train_sizes

        self.train_losses.append(total_loss / total_samples)
        self.train_acc.append(total_correct / total_samples)
        self.train_topk_acc.append(topk_acc_epoch_train)

        return total_loss / total_samples, total_correct / total_samples, topk_acc_epoch_train

    def val_epoch(
            self,
            val_loader,
            criteria,
            dataset_attributes,
            use_gpu,
            fp16
    ):
        self.model.eval()
        with torch.no_grad():
            n_val = dataset_attributes['n_val']
            # Initialization of variables
            loss_epoch_val = 0
            n_correct_val = 0
            n_correct_topk_val = defaultdict(int)
            n_correct_avgk_val = defaultdict(int)

            class_acc_dict = {
                'class_acc': defaultdict(int),
                'class_topk_acc': {k: defaultdict(int) for k in LIST_K},
                'class_avgk_acc': {k: defaultdict(int) for k in LIST_K}
            }

            list_val_proba = []
            list_val_labels = []

            for batch_idx, (batch_x_val, batch_y_val) in enumerate(tqdm(val_loader, desc='val', position=0)):
                if use_gpu:
                    batch_x_val, batch_y_val = batch_x_val.cuda(), batch_y_val.cuda()

                if fp16:
                    with torch.amp.autocast(device_type='cuda'):
                        batch_output_val = self.model(batch_x_val).logits
                        batch_proba = F.softmax(batch_output_val)
                else:
                    batch_output_val = self.model(batch_x_val).logits
                    batch_proba = F.softmax(batch_output_val)

                list_val_proba.append(batch_proba)
                list_val_labels.append(batch_y_val)

                loss_batch_val = criteria(batch_output_val, batch_y_val)
                loss_epoch_val += loss_batch_val.item()

                n_correct_val += (batch_y_val == torch.argmax(batch_output_val, dim=-1)).sum().item()
                update_correct_per_class(batch_proba, batch_y_val, class_acc_dict['class_acc'])

                for k in LIST_K:
                    n_correct_topk_val[k] += count_correct_topk(batch_output_val, batch_y_val, k).item()
                    update_correct_per_class_topk(batch_proba, batch_y_val, class_acc_dict['class_topk_acc'][k], k)

            val_probas = torch.cat(list_val_proba)
            val_labels = torch.cat(list_val_labels)

            flat_val_probas = val_probas.flatten()  # No need for torch.flatten
            sorted_probas, _ = torch.sort(flat_val_probas, descending=True)

            lmbda_val = {}
            for k in LIST_K:
                lmbda_val[k] = 0.5 * (sorted_probas[n_val * k - 1] + sorted_probas[n_val * k])
                n_correct_avgk_val[k] += count_correct_avgk(val_probas, val_labels, lmbda_val[k]).item()
                update_correct_per_class_avgk(val_probas, val_labels, class_acc_dict['class_avgk_acc'][k], lmbda_val[k])

            loss_epoch_val /= (batch_idx + 1)  # Avoid division by zero
            epoch_accuracy_val = n_correct_val / n_val

            topk_acc_epoch_val = {k: n_correct_topk_val[k] / n_val for k in LIST_K}
            avgk_acc_epoch_val = {k: n_correct_avgk_val[k] / n_val for k in LIST_K}

            for class_id, n_class_val in dataset_attributes['class2num_instances']['val'].items():
                class_acc_dict['class_acc'][class_id] /= n_class_val
                for k in LIST_K:
                    class_acc_dict['class_topk_acc'][k][class_id] /= n_class_val
                    class_acc_dict['class_avgk_acc'][k][class_id] /= n_class_val

            self.val_losses.append(loss_epoch_val)
            self.val_acc.append(epoch_accuracy_val)
            self.val_topk_acc.append(topk_acc_epoch_val)
            self.val_avgk_acc.append(avgk_acc_epoch_val)
            self.val_class_acc.append(class_acc_dict)

        return loss_epoch_val, epoch_accuracy_val, topk_acc_epoch_val, avgk_acc_epoch_val, lmbda_val

    def test_epoch(
            self,
            test_loader,
            criteria,
            dataset_attributes,
            lmbda,
            use_gpu,
            fp16
    ):
        print()  # Add a newline for visual separation in output
        self.model.eval()
        with torch.no_grad():
            n_test = test_loader.dataset.n_test
            # Initialization of variables
            loss_epoch_test = 0
            n_correct_test = 0
            n_correct_topk_test = defaultdict(int)
            n_correct_avgk_test = defaultdict(int)

            class_acc_dict = {
                'class_acc': defaultdict(int),
                'class_topk_acc': {k: defaultdict(int) for k in LIST_K},
                'class_avgk_acc': {k: defaultdict(int) for k in LIST_K}
            }

            for batch_idx, (batch_x_test, batch_y_test) in enumerate(tqdm(test_loader, desc='test', position=0)):
                if use_gpu:
                    batch_x_test, batch_y_test = batch_x_test.cuda(), batch_y_test.cuda()

                if fp16:
                    with torch.cuda.amp.autocast():
                        batch_output_test = self.model(batch_x_test)
                        batch_proba_test = F.softmax(batch_output_test, dim=1)
                else:
                    batch_output_test = self.model(batch_x_test)
                    batch_proba_test = F.softmax(batch_output_test, dim=1)

                loss_batch_test = criteria(batch_output_test, batch_y_test)
                loss_epoch_test += loss_batch_test.item()

                n_correct_test += (batch_y_test == torch.argmax(batch_output_test, dim=-1)).sum().item()
                update_correct_per_class(batch_proba_test, batch_y_test, class_acc_dict['class_acc'])

                for k in LIST_K:
                    n_correct_topk_test[k] += count_correct_topk(batch_output_test, batch_y_test, k).item()
                    n_correct_avgk_test[k] += count_correct_avgk(batch_proba_test, batch_y_test, lmbda[k]).item()
                    update_correct_per_class_topk(batch_proba_test, batch_y_test, class_acc_dict['class_topk_acc'][k],
                                                  k)
                    update_correct_per_class_avgk(batch_proba_test, batch_y_test, class_acc_dict['class_avgk_acc'][k],
                                                  lmbda[k])

            loss_epoch_test /= (batch_idx + 1)  # Avoid division by zero
            epoch_accuracy_test = n_correct_test / n_test

            topk_acc_epoch_test = {k: n_correct_topk_test[k] / n_test for k in LIST_K}
            avgk_acc_epoch_test = {k: n_correct_avgk_test[k] / n_test for k in LIST_K}

            for class_id, n_class_test in dataset_attributes['class2num_instances']['test'].items():
                class_acc_dict['class_acc'][class_id] /= n_class_test
                for k in LIST_K:
                    class_acc_dict['class_topk_acc'][k][class_id] /= n_class_test
                    class_acc_dict['class_avgk_acc'][k][class_id] /= n_class_test

        return loss_epoch_test, epoch_accuracy_test, topk_acc_epoch_test, avgk_acc_epoch_test, class_acc_dict

    def train(self, args, train_loader, val_loader, dataset_attributes, use_gpu):
        set_seed(args.seed, use_gpu=use_gpu)
        criteria = LabelSmoothingCrossEntropy(smoothing_factor=args.smoothing)

        if use_gpu:
            self.model.cuda()
            criteria.cuda()

        # TODO: Currently fixed optimizer & scheduler but will change later on
        optimizer = AdamW(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = LambdaLR(
            optimizer,
            lr_lambda=lambda current_step: lr_lambda(current_step, args.warmup_steps, args.epochs * len(train_loader))
        )
        scaler = GradScaler()
        # TODO: Re-org as json file for better control
        save_name = args.save_name.strip()
        save_dir = os.path.join(os.getcwd(), 'results', save_name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        lmbda_best_acc = None
        best_val_accuracy = float('-inf')

        # Training loop
        for epoch in tqdm(range(args.epochs), desc='epoch', position=0):
            t = time.time()
            # Train
            (
                loss_epoch_train,
                epoch_accuracy_train,
                topk_acc_epoch_train
            ) = self.train_epoch(
                train_loader,
                optimizer,
                scheduler,
                criteria,
                use_gpu,
                args.fp16,
                scaler
            )
            # Valid
            (
                loss_epoch_val,
                epoch_accuracy_val,
                topk_acc_epoch_val,
                avgk_acc_epoch_val,
                lmbda_val
            ) = self.val_epoch(
                val_loader,
                criteria,
                use_gpu
            )

            # save model with best val accuracy
            if epoch_accuracy_val > best_val_accuracy:
                best_val_accuracy = epoch_accuracy_val
                lmbda_best_acc = lmbda_val
                save(self.model, optimizer, epoch, os.path.join(save_dir, save_name + '_weights_best_acc.tar'))

            print()
            print(f'epoch {epoch} took {time.time() - t:.2f}')
            print(f'loss_train : {loss_epoch_train}')
            print(f'loss_val : {loss_epoch_val}')
            print(f'acc_train : {epoch_accuracy_train} / topk_acc_train : {topk_acc_epoch_train}')
            print(f'acc_val : {epoch_accuracy_val} / topk_acc_val : {topk_acc_epoch_val} / '
                  f'avgk_acc_val : {avgk_acc_epoch_val}')

