import torch
import os

from torch.optim import AdamW
from transformers import BertModel

from data_aug_for_pm.trainer.bert_trainer import Trainer as BertTrainer
from data_aug_for_pm.utils.load_config import ExperimentConfiguration
from data_aug_for_pm.augmenter.operator_mixup import apply_mixup


class Trainer(BertTrainer):
    def __init__(self, model, train_dataloader, val_dataloader, experiment: ExperimentConfiguration, working_dir: str):
        super().__init__(model, train_dataloader, val_dataloader, experiment, working_dir)
        self.optimizer = AdamW(self.model.parameters(), lr=0.01, eps=1e-8)
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.bert.to(self.device)

    def load_trained_model(self):
        if self.model_name == "cnn":
            self.model = get_cnn_model()
        elif self.model_name == "rnn":
            self.model = get_rnn_model()
        else:
            raise NotImplementedError("Unknown model name when loading model.")

        self.model.load_state_dict(torch.load(os.path.join(self.model_path, "model.pt")))
        self.model.to(self.device)
        self.model.eval()

    def move_to_cuda_and_get_model_output(self, batch, train=False):
        # left
        input_ids = batch["input_ids"].to(self.device)
        token_type_ids = batch["token_type_ids"].to(self.device)

        # right
        input_ids_2 = batch["input_ids_2"].to(self.device)
        token_type_ids_2 = batch["token_type_ids_2"].to(self.device)

        labels = batch["labels"].to(self.device)

        # for soft labels
        labels = labels.unsqueeze(1).repeat((1, 2))
        labels[:, 0] = 1 - labels[:, 0]

        # feed though bert
        with torch.no_grad():
            embedding_1 = self.bert(input_ids=input_ids, token_type_ids=token_type_ids).last_hidden_state
            embedding_2 = self.bert(input_ids=input_ids_2, token_type_ids=token_type_ids_2).last_hidden_state

        # online data augmentation
        if train and "mixup" in self.experiment.online_augmentation:
            embedding_1, embedding_2, labels = apply_mixup(embedding_1, embedding_2, labels)

        input = torch.concat([embedding_1, embedding_2], dim=1)

        if train and "mixda" in self.experiment.online_augmentation:
            aug_input_ids = batch["aug_input_ids"].to(self.device)
            aug_token_type_ids = batch["aug_token_type_ids"].to(self.device)
            aug_input_ids_2 = batch["aug_input_ids_2"].to(self.device)
            aug_token_type_ids_2 = batch["aug_token_type_ids_2"].to(self.device)

            with torch.no_grad():
                aug_embedding_1 = self.bert(input_ids=aug_input_ids, token_type_ids=aug_token_type_ids).last_hidden_state
                aug_embedding_2 = self.bert(input_ids=aug_input_ids_2, token_type_ids=aug_token_type_ids_2).last_hidden_state
                aug_input = torch.concat([aug_embedding_1, aug_embedding_2], dim=1)
                input = torch.concat([input, aug_input], dim=0)

        if train:
            result = self.model(input, mixda=True if "mixda" in self.experiment.online_augmentation else False)
        else:
            with torch.no_grad():
                result = self.model(input)

        return result, labels
