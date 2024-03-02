"""
Techniques: Synonym replacement, random deletion, random swap, random insertion.

The actual augmentation operations are from https://github.com/jasonwei20/eda_nlp
"""
from typing import List

import pandas as pd
import re
import random

from nltk.corpus import wordnet

import nltk

nltk.download('wordnet')
nltk.download('stopwords')  # not needed here but for ditto


def _apply_augmentation(df: 'pd.DataFrame', columns_to_augment: List[str], augmenting_func: 'function',
                        inflate_factor: int = 3) -> 'pd.DataFrame':
    not_augmented_cols = [i for i in df.columns.to_list() if i not in columns_to_augment]
    augmented_rows = {item: [] for item in df.columns.to_list()}

    for _, row in df.iterrows():
        for _ in range(inflate_factor - 1):
            for col in columns_to_augment:
                if row[col] is None:
                    augmented_rows[col].append(" ")
                else:
                    sentence = str(row[col]).split(" ")
                    augmented_sentence = augmenting_func(sentence, 0.1)
                    augmented_sentence = " ".join(augmented_sentence)
                    augmented_rows[col].append(augmented_sentence)

            for col in not_augmented_cols:
                augmented_rows[col].append(row[col])

    augmented_df = pd.concat([df, pd.DataFrame(augmented_rows)]).sample(frac=1).reset_index(drop=True)
    return augmented_df


def delete_random_words(df: 'pd.DataFrame', columns_to_augment: List[str], inflate_factor: int = 3) -> 'pd.DataFrame':
    return _apply_augmentation(
        df=df,
        columns_to_augment=columns_to_augment,
        augmenting_func=_random_deletion,
        inflate_factor=inflate_factor
    )


def replace_with_synonyms(df: 'pd.DataFrame', columns_to_augment: List[str], inflate_factor: int = 3) -> 'pd.DataFrame':
    return _apply_augmentation(
        df=df,
        columns_to_augment=columns_to_augment,
        augmenting_func=_synonym_replacement,
        inflate_factor=inflate_factor
    )


def swap_random_words(df: 'pd.DataFrame', columns_to_augment: List[str], inflate_factor: int = 3) -> 'pd.DataFrame':
    return _apply_augmentation(
        df=df,
        columns_to_augment=columns_to_augment,
        augmenting_func=_random_swap,
        inflate_factor=inflate_factor
    )


def insert_random_words(df: 'pd.DataFrame', columns_to_augment: List[str], inflate_factor: int = 3) -> 'pd.DataFrame':
    return _apply_augmentation(
        df=df,
        columns_to_augment=columns_to_augment,
        augmenting_func=_random_insertion,
        inflate_factor=inflate_factor
    )


def _random_insertion(words, p):
    n = max(1, int(p * len(words)))

    new_words = words.copy()
    for _ in range(n):
        _add_word(new_words)
    return new_words


def _add_word(new_words):
    synonyms = []
    counter = 0
    while len(synonyms) < 1:
        random_word = new_words[random.randint(0, len(new_words) - 1)]
        synonyms = _get_synonyms(random_word)
        counter += 1
        if counter >= 10:
            return
    random_synonym = synonyms[0]
    random_idx = random.randint(0, len(new_words) - 1)
    new_words.insert(random_idx, random_synonym)


def _random_swap(words, p):
    n = max(1, int(p * len(words)))

    new_words = words.copy()
    for _ in range(n):
        new_words = _swap_word(new_words)
    return new_words


def _swap_word(new_words):
    random_idx_1 = random.randint(0, len(new_words) - 1)
    random_idx_2 = random_idx_1
    counter = 0
    while random_idx_2 == random_idx_1:
        random_idx_2 = random.randint(0, len(new_words) - 1)
        counter += 1
        if counter > 3:
            return new_words
    new_words[random_idx_1], new_words[random_idx_2] = new_words[random_idx_2], new_words[random_idx_1]
    return new_words


def _random_deletion(words, p):
    # if there's only one word, don't delete it
    if len(words) == 1:
        return words

    # randomly delete words with probability p
    new_words = []
    for word in words:
        r = random.uniform(0, 1)
        if r > p:
            new_words.append(word)

    # if you end up deleting all words, just return a random word
    if len(new_words) == 0:
        rand_int = random.randint(0, len(words) - 1)
        return [words[rand_int]]

    return new_words


def _get_only_chars(line):
    """
    from eda repo
    """
    clean_line = ""

    line = line.replace("’", "")
    line = line.replace("'", "")
    line = line.replace("-", " ")
    line = line.replace("\t", " ")
    line = line.replace("\n", " ")
    line = line.lower()

    for char in line:
        if char in 'qwertyuiopasdfghjklzxcvbnm ':
            clean_line += char
        else:
            clean_line += ' '

    clean_line = re.sub(' +', ' ', clean_line)  # delete extra spaces
    if clean_line[0] == ' ':
        clean_line = clean_line[1:]
    return clean_line


def _synonym_replacement(words, p):
    n = max(1, int(p * len(words)))

    new_words = words.copy()
    random_word_list = list(set(words))
    random.shuffle(random_word_list)
    num_replaced = 0
    for random_word in random_word_list:
        synonyms = _get_synonyms(random_word)
        if len(synonyms) >= 1:
            synonym = random.choice(list(synonyms))
            new_words = [synonym if word == random_word else word for word in new_words]
            num_replaced += 1
        if num_replaced >= n:  # only replace up to n words
            break

    sentence = ' '.join(new_words)
    new_words = sentence.split(' ')

    return new_words


def _get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for l in syn.lemmas():
            synonym = l.name().replace("_", " ").replace("-", " ").lower()
            synonym = "".join([char for char in synonym if char in ' qwertyuiopasdfghjklzxcvbnm'])
            synonyms.add(synonym)
    if word in synonyms:
        synonyms.remove(word)
    return list(synonyms)
