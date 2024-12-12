"""
Custom code for Pl@ntNet-300K but using mobilevit v2 from HuggingFace
"""
import os
import time
import torch
from tqdm import tqdm
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from torch.nn import CrossEntropyLoss
from epoch import train_epoch, val_epoch, test_epoch


def train(model, train_loader, optimizer, scheduler, criteria, use_gpu, fp16, scaler):
    model.train()
    total_loss, total_correct  = 0, 0
    total_samples = 0

    for batch_idx, (batch_x_train, batch_y_train) in enumerate(tqdm(train_loader, desc='train', position=0)):
        if use_gpu:
            batch_x_train, batch_y_train = batch_x_train.cuda(), batch_y_train.cuda()
            model = model.cuda()

        optimizer.zero_grad()
        if fp16:
            with torch.cuda.amp.autocast():
                outputs = model(batch_x_train)
                loss = criteria(outputs, batch_y_train)

        # The gradient scaler will become no op if it detects the system is using fp32 instead
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if scheduler:
            scheduler.step()
        train_sizes = batch_x_train.size(0)
        total_loss += loss.item() * train_sizes
        total_correct += (outputs.logits.argmax(dim=-1) == batch_y_train).sum().item()
        total_samples += train_sizes

        # Get top_k accuracy
        for k in list_k:
            n_correct_topk_train[k] += count_correct_topk(scores=batch_output_train, labels=batch_y_train, k=k).item()

        return total_loss / total_samples, total_correct / total_samples




def pipi