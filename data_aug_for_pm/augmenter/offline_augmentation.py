from data_aug_for_pm.augmenter.operator_contextual_aug import apply_contextual_aug
from data_aug_for_pm.augmenter.operator_easy_data_augmentation import (
    replace_with_synonyms,
    insert_random_words,
    delete_random_words,
    swap_random_words,
)
from data_aug_for_pm.augmenter.operator_invda import apply_invda


def apply_offline_augmentation(
    df, columns_to_augment: list, offline_augmentation: list, experiment_configuration=None, global_configuration=None
):
    for operation in offline_augmentation:
        if operation == "synonym":
            df = replace_with_synonyms(df, columns_to_augment=columns_to_augment)
        elif operation == "swap_random":
            df = swap_random_words(df, columns_to_augment=columns_to_augment)
        elif operation == "insert_random":
            df = insert_random_words(df, columns_to_augment=columns_to_augment)
        elif operation == "delete_random":
            df = delete_random_words(df, columns_to_augment=columns_to_augment)
        elif operation == "invda":
            df = apply_invda(df, columns_to_augment, experiment_configuration, global_configuration)
        elif operation == "contextual":
            df = apply_contextual_aug(df, columns_to_augment)
        else:
            raise NotImplementedError("Unknown offline augmentation operation in yaml")

    return df
