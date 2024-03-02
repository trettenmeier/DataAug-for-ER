import torch
import numpy as np

from transformers import BertModel

from data_aug_for_pm.utils.load_config import ExperimentConfiguration
from data_aug_for_pm.augmenter.operator_mixup import apply_mixup


def get_model(experiment: ExperimentConfiguration):
    return Model(experiment=experiment)


class Model(torch.nn.Module):
    def __init__(self, experiment: ExperimentConfiguration):
        super(Model, self).__init__()
        self.experiment = experiment

        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.dropout = torch.nn.Dropout(p=0.1)
        self.fc = torch.nn.Linear(in_features=self.bert.config.hidden_size, out_features=2)

    def forward(self, input_ids, token_type_ids, labels=None, mixda=False):
        """
        The labels are only needed if we apply online augmentation, e.g. mixup.
        """
        embeddings = self.bert.embeddings(input_ids=input_ids, token_type_ids=token_type_ids)

        if "mixup" in self.experiment.online_augmentation and labels is not None:
            embeddings, labels = apply_mixup(embeddings, labels)

        encoded = self.bert.encoder(embeddings)
        output = self.bert.pooler(encoded.last_hidden_state)

        output = self.dropout(output)

        if mixda:
            batch_size = int(len(output) / 2)
            enc1 = output[:batch_size]
            enc2 = output[batch_size:]

            aug_lam = np.random.beta(0.8, 0.8)
            output = enc1 * aug_lam + enc2 * (1.0 - aug_lam)

        if labels is not None:
            return self.fc(output), labels
        return self.fc(output)
