import random
import logging
import torch
import pandas as pd

from typing import List
from tqdm import tqdm

from data_aug_for_pm.augmenter.invda.train_t5 import main as train_t5
from data_aug_for_pm.augmenter.invda.generate import generate
from data_aug_for_pm.utils.load_config import ExperimentConfiguration, GlobalConfiguration


# download spacy beforehand: python -m spacy download en_core_web_sm


def apply_invda(df: pd.DataFrame, columns_to_augment: List[str], experiment_configuration: ExperimentConfiguration,
                global_config: GlobalConfiguration):
    # train generative model
    model, tokenizer = train_t5(experiment_configuration=experiment_configuration, global_configuration=global_config)

    # model and tokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    number_of_rows = df.shape[0]

    df = df.reset_index(drop=True)
    logging.info(f"Shape of df before invda: {df.shape}")

    chunk_size = 128
    end = df.shape[0]

    inflate_factor = 2
    augmented_dfs = [df.copy(deep=True) for _ in range(inflate_factor)]

    for i in range(len(augmented_dfs)):
        for start in tqdm(range(0, end+1, chunk_size), total=int(number_of_rows/chunk_size)):
            end_chunk = min(start + chunk_size - 1, end)

            col_to_augment = random.choice(columns_to_augment)
            sentence = df[col_to_augment].iloc[start:end_chunk+1].tolist()
            aug_value_list = generate(model=model, tokenizer=tokenizer, device=device, sentence=sentence,
                                      experiment_configuration=experiment_configuration)
            if aug_value_list == "empty":
                continue
            augmented_dfs[i].loc[start:end_chunk, col_to_augment] = aug_value_list

    df_aug = pd.concat([i for i in augmented_dfs])
    df = pd.concat([df, df_aug]).reset_index(drop=True)

    del model
    logging.info(f"Shape of df after invda: {df.shape}")
    return df
