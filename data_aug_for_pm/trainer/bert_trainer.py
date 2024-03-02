import torch
import random
import numpy as np
import time
import datetime
import logging
import os

from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn import CrossEntropyLoss
from sklearn.metrics import f1_score
from scipy.special import softmax

from data_aug_for_pm.utils.load_config import ExperimentConfiguration
from data_aug_for_pm.utils.early_stopping import EarlyStopper, StopTrainingWhenTrainLossIsNearZero
from data_aug_for_pm.models.bert import get_model


class Trainer:
    def __init__(self, model, train_dataloader, val_dataloader, experiment: ExperimentConfiguration, working_dir: str):
        seed_val = 42

        random.seed(seed_val)
        np.random.seed(seed_val)
        torch.manual_seed(seed_val)
        torch.cuda.manual_seed_all(seed_val)

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            logging.info(f'Using GPU:, {torch.cuda.get_device_name(0)}')
        else:
            logging.info('No GPU available')
            self.device = torch.device("cpu")

        self.experiment = experiment
        self.model_name = experiment.model
        self.model = model
        self.model.to(self.device)

        self.early_stopper = EarlyStopper()
        self.stop_training = False

        self.model_path = os.path.join(working_dir, "models", self.model_name, "trained_model")
        os.makedirs(self.model_path, exist_ok=True)

        self.epochs = experiment.epochs
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = AdamW(self.model.parameters(), lr=2e-5, eps=1e-8)

        self.reduce_lr_on_plateau = ReduceLROnPlateau(self.optimizer, 'min', factor=0.5, patience=5, verbose=True)

        self.training_stats = []
        self.best_val_f1 = -1

    def load_trained_model(self):
        self.model = get_model(self.experiment)
        self.model.load_state_dict(torch.load(os.path.join(self.model_path, "model.pt")))
        self.model.to(self.device)
        self.model.eval()

    def move_to_cuda_and_get_model_output(self, batch, train=False):
        input_ids = batch["input_ids"].to(self.device)
        token_type_ids = batch["token_type_ids"].to(self.device)
        labels = batch["labels"].to(self.device)

        # for soft labels
        labels = labels.unsqueeze(1).repeat((1, 2))
        labels[:, 0] = 1 - labels[:, 0]

        # ditto aug
        mixda = "mixda" in self.experiment.online_augmentation
        if train and mixda:
            aug_input_ids = batch["aug_input_ids"].to(self.device)
            aug_token_type_ids = batch["aug_token_type_ids"].to(self.device)

            input_ids = torch.concat([input_ids, aug_input_ids], dim=0)
            token_type_ids = torch.concat([token_type_ids, aug_token_type_ids], dim=0)

        # forward pass
        if train:
            result, labels = self.model(input_ids=input_ids, token_type_ids=token_type_ids, labels=labels, mixda=mixda)
        else:
            with torch.no_grad():
                result = self.model(input_ids=input_ids, token_type_ids=token_type_ids)

        return result, labels

    def train(self):
        self.training_stats = []
        training_start_time = time.time()
        avg_train_loss = 0

        loss = CrossEntropyLoss()

        for epoch in range(0, self.epochs):
            logging.info(f'======== Epoch {epoch + 1} / {self.epochs} ========')
            t0 = time.time()
            self.model.train()

            if StopTrainingWhenTrainLossIsNearZero.training_loss_is_near_zero(avg_train_loss) and avg_train_loss != 0:
                logging.info("")
                logging.info("Loading best model.")
                self.load_trained_model()
                self.model.eval()
                logging.info(f"Total training took {self.format_time(time.time() - training_start_time)}")
                break

            total_train_loss = 0

            for step, batch in enumerate(self.train_dataloader):
                if step % 40 == 0 and not step == 0:
                    elapsed = self.format_time(time.time() - t0)
                    logging.info(f'Step {step}  of  {len(self.train_dataloader)}. Elapsed: {elapsed}')

                self.model.zero_grad()
                self.optimizer.zero_grad()

                result, labels = self.move_to_cuda_and_get_model_output(batch, train=True)

                output = loss(result, labels)
                output.backward()
                total_train_loss += output.item()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                self.optimizer.step()
                loss.zero_grad()

                # validate every n steps
                if step % 10000 == 0 and not step == 0:
                    self.validation(epoch, 0, self.format_time(time.time() - t0))
                    self.model.train()

            avg_train_loss = total_train_loss / len(self.train_dataloader)
            training_time = self.format_time(time.time() - t0)

            logging.info("")
            logging.info(f"  Average training loss: {avg_train_loss}")
            logging.info(f"  Training epoch took: {training_time}")
            logging.info("")
            logging.info("Running Validation...")
            avg_val_loss = self.validation(epoch, avg_train_loss, training_time)

            self.reduce_lr_on_plateau.step(avg_val_loss)

            if self.stop_training:
                logging.info("early stopping.")
                break

        logging.info("")
        logging.info("Loading best model.")
        self.load_trained_model()
        logging.info(f"Total training took {self.format_time(time.time() - training_start_time)}")

    def validation(self, epoch, avg_train_loss, training_time) -> float:
        t0 = time.time()
        total_eval_loss = 0
        all_logits = []
        all_labels = []

        self.model.eval()
        from torch.nn import CrossEntropyLoss
        loss = CrossEntropyLoss()

        for batch in self.val_dataloader:
            result, labels = self.move_to_cuda_and_get_model_output(batch, train=False)

            output = loss(result, labels)

            total_eval_loss += output.item()

            loss.zero_grad()
            raw_logits = result.detach().to('cpu').numpy()
            softmaxed_logits = softmax(raw_logits, axis=1)

            all_logits.extend(softmaxed_logits[:, 1])

            # create integer labels for f1 score
            int_labels = labels[:, 1].to('cpu').round().to(torch.int64)
            all_labels.extend(int_labels)

        f1 = self.compute_f1(all_logits, all_labels)
        logging.info(f"  Validation F1: {f1}")

        avg_val_loss = total_eval_loss / len(self.val_dataloader)
        validation_time = self.format_time(time.time() - t0)

        logging.info(f"  Validation Loss: {avg_val_loss}")
        logging.info(f"  Validation took: {validation_time}")

        self.training_stats.append({
            'epoch': epoch + 1,
            'Training Loss': avg_train_loss,
            'Valid. Loss': avg_val_loss,
            'Valid. F1.': f1,
            'Training Time': training_time,
            'Validation Time': validation_time
        })

        # save best model
        if f1 > self.best_val_f1:
            self.best_val_f1 = f1
            torch.save(self.model.state_dict(), os.path.join(self.model_path, "model.pt"))

        # early stopping
        self.stop_training = self.early_stopper.early_stop(avg_val_loss)

        return avg_val_loss

    def format_time(self, elapsed):
        elapsed_rounded = int(round((elapsed)))
        return str(datetime.timedelta(seconds=elapsed_rounded))

    def compute_f1(self, probability_pos, labels):
        thresholds = np.linspace(0, 1, num=50)

        def calc_func(e: float):
            predictions = [1 if x > e else 0 for x in probability_pos]
            return f1_score(labels, predictions)

        results = np.array([calc_func(thresh) for thresh in thresholds])

        return max(results)

    def evaluate(self, test_dataloader):
        self.load_trained_model()
        self.model.eval()

        predictions, true_labels = [], []

        for batch in test_dataloader:
            result, labels = self.move_to_cuda_and_get_model_output(batch, train=False)

            logits = result.detach().cpu().numpy()
            label_ids = labels.to('cpu').numpy()

            predictions.append(logits)
            true_labels.append(label_ids)

        pred_flat = list(np.concatenate(predictions))
        pred_flat = softmax(pred_flat, axis=1)

        true_labels_flat = list(np.concatenate(true_labels))
        true_labels_flat = [int(round(i[1])) for i in true_labels_flat]

        return pred_flat, true_labels_flat
