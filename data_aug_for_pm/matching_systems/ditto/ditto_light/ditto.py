import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np
import sklearn.metrics as metrics
import argparse

from .dataset import DittoDataset
from torch.utils import data
from transformers import AutoModel, AdamW, get_linear_schedule_with_warmup
from tensorboardX import SummaryWriter

from data_aug_for_pm.utils.early_stopping import EarlyStopper, StopTrainingWhenTrainLossIsNearZero
from data_aug_for_pm.augmenter.operator_mixup import apply_mixup

# from apex import amp

lm_mp = {'roberta': 'roberta-base',
         'distilbert': 'distilbert-base-uncased'}


class DittoModel(nn.Module):
    """A baseline model for EM."""

    def __init__(self, device='cuda', lm='roberta', alpha_aug=0.8, experiment=None):
        super().__init__()
        if lm in lm_mp:
            self.bert = AutoModel.from_pretrained(lm_mp[lm])
        else:
            self.bert = AutoModel.from_pretrained(lm)

        self.device = device
        self.alpha_aug = alpha_aug
        self.experiment = experiment

        # linear layer
        hidden_size = self.bert.config.hidden_size
        self.fc = torch.nn.Linear(hidden_size, 2)

    def forward(self, x1, x2=None, labels=None):
        """Encode the left, right, and the concatenation of left+right.

        Args:
            x1 (LongTensor): a batch of ID's
            x2 (LongTensor, optional): a batch of ID's (augmented)

        Returns:
            Tensor: binary prediction
        """
        x1 = x1.to(self.device)  # (batch_size, seq_len)
        if x2 is not None:
            # MixDA
            x2 = x2.to(self.device)  # (batch_size, seq_len)
            enc = self.bert(torch.cat((x1, x2)))[0][:, 0, :]
            batch_size = len(x1)
            enc1 = enc[:batch_size]  # (batch_size, emb_size)
            enc2 = enc[batch_size:]  # (batch_size, emb_size)

            aug_lam = np.random.beta(self.alpha_aug, self.alpha_aug)
            enc = enc1 * aug_lam + enc2 * (1.0 - aug_lam)
        else:
            embeddings = self.bert.embeddings(input_ids=x1)

            if "mixup" in self.experiment.online_augmentation and labels is not None:
                left, right, labels = apply_mixup(embeddings, labels)

            head_mask = [None] * self.bert.config.num_hidden_layers
            attention_mask = torch.ones((len(embeddings), len(embeddings[0]))).to(self.device)

            enc = self.bert.transformer(
                x=embeddings,
                attn_mask=attention_mask,
                head_mask=head_mask,
                return_dict=True
            )
            enc = enc[0][:, 0, :]
            # enc = self.bert(x1)[0][:, 0, :]

        if labels is not None:
            return self.fc(enc), labels
        return self.fc(enc)


def evaluate(model, iterator, threshold=None, return_probs=False):
    """Evaluate a model on a validation/test dataset

    Args:
        model (DMModel): the EM model
        iterator (Iterator): the valid/test dataset iterator
        threshold (float, optional): the threshold on the 0-class

    Returns:
        float: the F1 score
        float (optional): if threshold is not provided, the threshold
            value that gives the optimal F1
    """
    all_p = []
    all_y = []
    all_probs = []
    with torch.no_grad():
        for batch in iterator:
            x, y = batch
            logits = model(x)
            probs = logits.softmax(dim=1)[:, 1]
            all_probs += probs.cpu().numpy().tolist()

            # for soft labels
            y = y.unsqueeze(1).repeat((1, 2))
            y[:, 0] = 1 - y[:, 0]

            all_y += y.cpu().numpy().tolist()

    if return_probs:
        return all_probs

    # here we need all y as integer labels for F1 score
    int_labels = [round(int(i[1])) for i in all_y]

    if threshold is not None:
        pred = [1 if p > threshold else 0 for p in all_probs]
        f1 = metrics.f1_score(int_labels, pred)
        return f1
    else:
        best_th = 0.5
        f1 = 0.0  # metrics.f1_score(all_y, all_p)

        for th in np.arange(0.0, 1.0, 0.05):
            pred = [1 if p > th else 0 for p in all_probs]
            new_f1 = metrics.f1_score(int_labels, pred)
            if new_f1 > f1:
                f1 = new_f1
                best_th = th

        return f1, best_th


def train_step(train_iter, model, optimizer, scheduler, hp, valid_iter, best_dev_f1, epoch):
    """Perform a single training step

    Args:
        train_iter (Iterator): the train data loader
        model (DMModel): the model
        optimizer (Optimizer): the optimizer (Adam or AdamW)
        scheduler (LRScheduler): learning rate scheduler
        hp (Namespace): other hyper-parameters (e.g., fp16)

    Returns:
        None
    """
    criterion = nn.CrossEntropyLoss()
    # criterion = nn.MSELoss()

    running_loss = 0.0
    avg_training_loss = 0.0

    for i, batch in enumerate(train_iter):
        optimizer.zero_grad()

        if len(batch) == 2:
            x, y = batch

            # for soft labels
            y = y.unsqueeze(1).repeat((1, 2))
            y[:, 0] = 1 - y[:, 0]

            prediction, y = model(x, labels=y)
        else:
            x1, x2, y = batch

            # for soft labels
            y = y.unsqueeze(1).repeat((1, 2))
            y[:, 0] = 1 - y[:, 0]
            prediction = model(x1, x2)

        loss = criterion(prediction, y.to(model.device))

        if hp.fp16:
            #     with amp.scale_loss(loss, optimizer) as scaled_loss:
            #         scaled_loss.backward()
            raise NotImplementedError()
        else:
            loss.backward()
        optimizer.step()
        scheduler.step()
        if i % 10 == 0:  # monitoring
            print(f"step: {i} / {len(train_iter)}, loss: {loss.item()}")

        running_loss += loss.item()
        del loss

        # validate every n steps
        if i % 10000 == 0 and not i == 0:
            model.eval()
            valid_f1, _ = evaluate(model, valid_iter)
            print(f"inter-epoch validation. F1: {valid_f1}")
            if valid_f1 > best_dev_f1:
                best_dev_f1 = valid_f1
                if hp.save_model:
                    # create the directory if not exist
                    directory = os.path.join(hp.logdir, hp.task)
                    if not os.path.exists(directory):
                        os.makedirs(directory)

                    # save the checkpoints for each component
                    ckpt_path = os.path.join(hp.logdir, hp.task, 'model.pt')
                    ckpt = {'model': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'scheduler': scheduler.state_dict(),
                            'epoch': epoch}
                    torch.save(ckpt, ckpt_path)
            model.train()

        avg_training_loss = running_loss / len(train_iter)

    return best_dev_f1, avg_training_loss


def train(trainset, validset, testset, run_tag, hp, experiment):
    """Train and evaluate the model

    Args:
        trainset (DittoDataset): the training set
        validset (DittoDataset): the validation set
        testset (DittoDataset): the test set
        run_tag (str): the tag of the run
        hp (Namespace): Hyper-parameters (e.g., batch_size,
                        learning rate, fp16)

    Returns:
        None
    """
    padder = trainset.pad
    # create the DataLoaders
    train_iter = data.DataLoader(dataset=trainset,
                                 batch_size=hp.batch_size,
                                 shuffle=True,
                                 num_workers=0,
                                 collate_fn=padder)
    valid_iter = data.DataLoader(dataset=validset,
                                 batch_size=hp.batch_size * 16,
                                 shuffle=False,
                                 num_workers=0,
                                 collate_fn=padder)
    test_iter = data.DataLoader(dataset=testset,
                                batch_size=hp.batch_size * 16,
                                shuffle=False,
                                num_workers=0,
                                collate_fn=padder)

    # initialize model, optimizer, and LR scheduler
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = DittoModel(device=device,
                       lm=hp.lm,
                       alpha_aug=hp.alpha_aug,
                       experiment=experiment)
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=hp.lr)

    # if hp.fp16:
    # model, optimizer = amp.initialize(model, optimizer, opt_level='O2')
    num_steps = (len(trainset) // hp.batch_size) * hp.n_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=num_steps)

    # logging with tensorboardX
    writer = SummaryWriter(log_dir=hp.logdir)

    best_dev_f1 = best_test_f1 = 0.0
    epochs_trained = 0
    avg_train_loss = 0.0

    early_stopper = EarlyStopper()
    ckpt = {"model": model.state_dict()}  # saved so we have that "model" key and are able access even if we end up not saving the model here due to intra-epoch validation
    for epoch in range(1, hp.n_epochs + 1):
        # train
        model.train()

        if avg_train_loss == 0.0 or not StopTrainingWhenTrainLossIsNearZero.training_loss_is_near_zero(avg_train_loss):
            best_dev_f1, avg_train_loss = train_step(train_iter, model, optimizer, scheduler, hp, valid_iter, best_dev_f1, epoch)

            # eval
            model.eval()
            dev_f1, th = evaluate(model, valid_iter)
            test_f1 = evaluate(model, test_iter, threshold=th)

            if dev_f1 > best_dev_f1:
                best_dev_f1 = dev_f1
                best_test_f1 = test_f1
                if hp.save_model:
                    # create the directory if not exist
                    directory = os.path.join(hp.logdir, hp.task)
                    if not os.path.exists(directory):
                        os.makedirs(directory)

                    # save the checkpoints for each component
                    ckpt_path = os.path.join(hp.logdir, hp.task, 'model.pt')
                    ckpt['model'] = model.state_dict()
                    ckpt['optimizer'] = optimizer.state_dict()
                    ckpt['scheduler'] = scheduler.state_dict()
                    ckpt['epoch'] = epoch
                    torch.save(ckpt, ckpt_path)

            print(f"epoch {epoch}: val_f1={dev_f1}, test_f1={test_f1}, best_test_f1={best_test_f1}")

            # logging
            scalars = {'f1': dev_f1,
                       't_f1': test_f1}
            writer.add_scalars(run_tag, scalars, epoch)

            epochs_trained = epoch
            if early_stopper.early_stop(1 - dev_f1):
                break

    writer.close()

    # evaluate with best model on test set after training is finished
    model.load_state_dict(ckpt["model"])
    all_probs = evaluate(model, test_iter, return_probs=True)
    return all_probs, epochs_trained
