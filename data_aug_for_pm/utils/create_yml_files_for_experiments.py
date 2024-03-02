import os
import yaml

from dataclasses import dataclass


@dataclass
class Dataset:
    name: str
    path_to_train_set: str
    path_to_val_set: str
    path_to_test_set: str


DATASETS = [
    Dataset(
        name="abt_buy",
        path_to_train_set="data/raw/abt_buy/train.csv",
        path_to_val_set="data/raw/abt_buy/valid.csv",
        path_to_test_set="data/raw/abt_buy/test.csv",
    ),
    Dataset(
        name="amazon_google",
        path_to_train_set="data/raw/amazon_google/train.csv",
        path_to_val_set="data/raw/amazon_google/valid.csv",
        path_to_test_set="data/raw/amazon_google/test.csv",
    ),
    Dataset(
        name="walmart_amazon",
        path_to_train_set="data/raw/walmart_amazon/train.csv",
        path_to_val_set="data/raw/walmart_amazon/valid.csv",
        path_to_test_set="data/raw/walmart_amazon/test.csv",
    ),
    Dataset(
        name="markt_pilot",
        path_to_train_set="data/raw/markt_pilot_dataset/markt_pilot_dataset_train.parquet",
        path_to_val_set="data/raw/markt_pilot_dataset/markt_pilot_dataset_val.parquet",
        path_to_test_set="data/raw/markt_pilot_dataset/markt_pilot_dataset_test.parquet",
    ),
    Dataset(
        name="markt_pilot",
        path_to_train_set="data/raw/markt_pilot_dataset/l_markt_pilot_dataset_train.parquet",
        path_to_val_set="data/raw/markt_pilot_dataset/l_markt_pilot_dataset_val.parquet",
        path_to_test_set="data/raw/markt_pilot_dataset/markt_pilot_dataset_test.parquet",
    ),
    Dataset(
        name="markt_pilot",
        path_to_train_set="data/raw/markt_pilot_dataset/m_markt_pilot_dataset_train.parquet",
        path_to_val_set="data/raw/markt_pilot_dataset/m_markt_pilot_dataset_val.parquet",
        path_to_test_set="data/raw/markt_pilot_dataset/markt_pilot_dataset_test.parquet",
    ),
    Dataset(
        name="markt_pilot",
        path_to_train_set="data/raw/markt_pilot_dataset/s_markt_pilot_dataset_train.parquet",
        path_to_val_set="data/raw/markt_pilot_dataset/s_markt_pilot_dataset_val.parquet",
        path_to_test_set="data/raw/markt_pilot_dataset/markt_pilot_dataset_test.parquet",
    ),
    Dataset(
        name="wdc",
        path_to_train_set="data/raw/wdc_lspm/all_train/all_train_xlarge.json.gz",
        path_to_val_set="data/raw/wdc_lspm/all_valid/all_valid_xlarge.csv",
        path_to_test_set="data/raw/wdc_lspm/all_gs.json.gz",
    ),
    Dataset(
        name="wdc",
        path_to_train_set="data/raw/wdc_lspm/all_train/all_train_large.json.gz",
        path_to_val_set="data/raw/wdc_lspm/all_valid/all_valid_large.csv",
        path_to_test_set="data/raw/wdc_lspm/all_gs.json.gz",
    ),
    Dataset(
        name="wdc",
        path_to_train_set="data/raw/wdc_lspm/all_train/all_train_medium.json.gz",
        path_to_val_set="data/raw/wdc_lspm/all_valid/all_valid_medium.csv",
        path_to_test_set="data/raw/wdc_lspm/all_gs.json.gz",
    ),
    Dataset(
        name="wdc",
        path_to_train_set="data/raw/wdc_lspm/all_train/all_train_small.json.gz",
        path_to_val_set="data/raw/wdc_lspm/all_valid/all_valid_small.csv",
        path_to_test_set="data/raw/wdc_lspm/all_gs.json.gz",
    ),
]

MODELS = ["bert", "ditto"]

OFFLINE_AUGMENTATION = [
    "synonym",
    "swap_random",
    "insert_random",
    "delete_random",
    "invda",
    "contextual",
    "network",
    "noising_disjoint",
    "change_numbers",
]
ONLINE_AUGMENTATION = ["mixda", "mixup"]
NO_AUGMENTATION = ["no_aug"]


def create_yml(dataset: Dataset, model: str, augmentation: str, is_offline: bool, path):
    yml_data = {
        "model": model,
        "dataset": dataset.name,
        "path_to_train_set": dataset.path_to_train_set,
        "path_to_val_set": dataset.path_to_val_set,
        "path_to_test_set": dataset.path_to_test_set,
        "batch_size": 32,
        "max_string_len": 1000,
        "max_input_length": 256 if model == "ditto" else 128,
        "epochs": 30,
    }

    if is_offline:
        if augmentation == "no_aug":
            yml_data["offline_augmentation"] = []
        else:
            yml_data["offline_augmentation"] = [augmentation]
        yml_data["online_augmentation"] = []
    else:
        yml_data["offline_augmentation"] = []
        yml_data["online_augmentation"] = [augmentation]

    def get_subset_size(path):
        if "wdc" in path and "_xlarge" in path:
            return "_xlarge"
        if "wdc" in path and "_large" in path:
            return "_large"
        if "wdc" in path and "_medium" in path:
            return "_medium"
        if "wdc" in path and "_small" in path:
            return "_small"
        if "markt_pilot" in path and "l_" in path:
            return "_l"
        if "markt_pilot" in path and "m_" in path:
            return "_m"
        if "markt_pilot" in path and "s_" in path:
            return "_s"
        if "markt_pilot_dataset_train" in path:
            return "_full"

        return ""

    filename = f"{dataset.name}{get_subset_size(dataset.path_to_train_set)}_{model}_{augmentation}.yml"

    with open(os.path.join(path, filename), "w") as file:
        yaml.dump(yml_data, file)


def main():
    path = os.path.join("..", "experiments")
    os.makedirs(path, exist_ok=True)

    for dataset in DATASETS:
        for model in MODELS:
            for offline_augmentation in OFFLINE_AUGMENTATION:
                create_yml(dataset, model, offline_augmentation, is_offline=True, path=path)

            for online_augmentation in ONLINE_AUGMENTATION:
                create_yml(dataset, model, online_augmentation, is_offline=False, path=path)

            for no_aug in NO_AUGMENTATION:
                create_yml(dataset, model, no_aug, is_offline=True, path=path)

    print(
        "Amount of experiments: ",
        len(DATASETS) * len(MODELS) * (len(OFFLINE_AUGMENTATION) + len(ONLINE_AUGMENTATION) + len(NO_AUGMENTATION)),
    )


if __name__ == "__main__":
    main()
