import torch

import pandas as pd
import transformers

from transformers import BertTokenizer
from tqdm import tqdm
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Dataset

from data_aug_for_pm.utils.load_config import ExperimentConfiguration
from data_aug_for_pm.augmenter.mixda import MixDA


class MarktPilotBertLoader:
    def __init__(self, df_train, df_val, df_test, experiment: ExperimentConfiguration):
        self.df_train = df_train
        self.df_val = df_val
        self.df_test = df_test
        self.experiment = experiment

    def get_train_loader(self):
        client_sentences, page_sentences, labels = self._get_sentences_and_labels(self.df_train)
        dataset = self._get_torch_dataset(client_sentences, page_sentences, labels, train_dataset=True)

        return DataLoader(
            dataset,
            sampler=RandomSampler(dataset),
            batch_size=self.experiment.batch_size
        )

    def get_val_loader(self):
        client_sentences, page_sentences, labels = self._get_sentences_and_labels(self.df_val)
        dataset = self._get_torch_dataset(client_sentences, page_sentences, labels)

        return DataLoader(
            dataset,
            sampler=RandomSampler(dataset),
            batch_size=self.experiment.batch_size
        )

    def get_test_loader(self):
        client_sentences, page_sentences, labels = self._get_sentences_and_labels(self.df_test)
        dataset = self._get_torch_dataset(client_sentences, page_sentences, labels)

        return DataLoader(
            dataset,
            sampler=SequentialSampler(dataset),
            batch_size=self.experiment.batch_size
        )

    def _get_sentences_and_labels(self, df):
        def page_concat(x):
            return " ".join([
                (x['found_brand']),
                (x['found_number']),
                (x['found_name']),
                (x['found_group']),
                (x['found_description']),
            ])

        def client_concat(x):
            return " ".join([
                (x['searched_brand']),
                (x['searched_number']),
                (x['searched_name']),
                (x['searched_group']),
                (x['searched_description']),
            ])

        df["page_concat"] = df.apply(page_concat, axis=1)
        df["client_concat"] = df.apply(client_concat, axis=1)

        # cap length
        def cap_len_page(x):
            return x["page_concat"][:self.experiment.max_string_len]

        def cap_len_client(x):
            return x["client_concat"][:self.experiment.max_string_len]

        df["page_concat"] = df.apply(cap_len_page, axis=1)
        df["client_concat"] = df.apply(cap_len_client, axis=1)

        # get lists of sentences and labels
        page_sentences = df["page_concat"].values
        client_sentences = df["client_concat"].values
        labels = df.label.astype(float).values

        return client_sentences, page_sentences, labels

    def _tokenize(self, client_sentences, page_sentences):
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

        input_ids = []
        token_type_ids = []

        transformers.logging.set_verbosity_error()

        for sent in tqdm(zip(page_sentences, client_sentences), total=len(client_sentences)):
            encoded_dict = tokenizer(
                text=sent[1],
                text_pair=sent[0],
                add_special_tokens=True,
                return_attention_mask=False,
                return_tensors='pt',
                max_length=self.experiment.max_input_length,
                padding='max_length',
                truncation=True
            )

            input_ids.append(torch.squeeze(encoded_dict['input_ids']))
            token_type_ids.append(torch.squeeze(encoded_dict["token_type_ids"]))

        transformers.utils.logging.set_verbosity_warning()

        return input_ids, token_type_ids

    def _get_torch_dataset(self, client_sentences, page_sentences, labels, train_dataset=False):
        input_ids, token_type_ids = self._tokenize(client_sentences, page_sentences)

        df = pd.DataFrame({
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "labels": labels,
            "left_sentences": client_sentences,
            "right_sentences": page_sentences
        })

        return CustomDataset(df, self.experiment, train_dataset=train_dataset)


class CustomDataset(Dataset):
    def __init__(self, df: pd.DataFrame, experiment: ExperimentConfiguration, train_dataset=False):
        """
        train_dataset decides if augmentation will be applied or not, independently if a augmentation-operator is chosen.
        """
        self.df = df
        self.experiment = experiment
        self.train_dataset = train_dataset
        self.augmenter = MixDA()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        if not (self.train_dataset and "mixda" in self.experiment.online_augmentation):
            return self.df.loc[idx].drop(columns=["left_sentences", "right_sentences"]).to_dict()
        else:
            return_value = self.df.loc[idx].to_dict()

            left_sentence = return_value["left_sentences"]
            right_sentence = return_value["right_sentences"]

            left_aug = self.augmenter.augment_sent(left_sentence)
            right_aug = self.augmenter.augment_sent(right_sentence)

            encoded_dict = self.tokenizer(
                text=left_aug,
                text_pair=right_aug,
                add_special_tokens=True,
                return_attention_mask=False,
                return_tensors='pt',
                max_length=self.experiment.max_input_length,
                padding='max_length',
                truncation=True
            )

            return_value["aug_input_ids"] = torch.squeeze(encoded_dict["input_ids"])
            return_value["aug_token_type_ids"] = torch.squeeze(encoded_dict["token_type_ids"])

            del return_value["left_sentences"]
            del return_value["right_sentences"]

            return return_value
