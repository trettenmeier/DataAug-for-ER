from torch.cuda import is_available as is_cuda_available
from typing import List
from nlpaug.augmenter.word.context_word_embs import ContextualWordEmbsAug
from tqdm import tqdm
import pandas as pd
import random
import logging


def apply_contextual_aug(df: pd.DataFrame, columns_to_augment: List[str], inflate_factor: int = 2) -> pd.DataFrame:
    device = "cuda" if is_cuda_available() else "cpu"
    augmenter = ContextualWordEmbsAug(model_type="bert", action="substitute", device=device, aug_min=1)
    
    augmented_dfs = [df.copy(deep=True) for _ in range(inflate_factor)]

    for i in range(len(augmented_dfs)):
        logging.info("applying contextual augmentations")
        for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
            col_to_augment = random.choice(columns_to_augment)
            sentence = str(df[col_to_augment].loc[idx])
            aug_sentence = augmenter.augment(sentence, n=1)
            if type(aug_sentence) == str:
                augmented_dfs[i].loc[idx, col_to_augment] = aug_sentence
            else:
                augmented_dfs[i].loc[idx, col_to_augment] = aug_sentence[0]

    del augmenter
    df_aug = pd.concat([i for i in augmented_dfs])
    df = pd.concat([df, df_aug]).reset_index(drop=True)

    return df
