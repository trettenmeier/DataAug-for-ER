import luigi
import os
import hashlib
import pandas as pd

from data_aug_for_pm.tasks.base import LuigiBaseTask
from data_aug_for_pm.utils.load_config import load_global_config, load_config
from data_aug_for_pm.utils.ditto import write_wdc_data_in_ditto_format
from data_aug_for_pm.augmenter.offline_augmentation import apply_offline_augmentation
from data_aug_for_pm.utils.wdc_dataset import set_datatypes_and_limit_string_length


class PreprocessWDCTrainDataTask(LuigiBaseTask):
    experiment_name = luigi.Parameter()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.global_config = load_global_config()
        self.experiment = load_config(self.experiment_name)

        # the relevant fields of the config get hashed, to detect unchanged fields in case of a rerun and save
        # computation time
        relevant_fields = " ".join([
            self.experiment.path_to_train_set,
            self.experiment.offline_augmentation.__str__()
        ])
        self.filename = hashlib.md5(relevant_fields.encode()).hexdigest()
        self.output_path = os.path.join(self.global_config.working_dir, "data", "intermediate", "load_wdc_data")

    def run(self) -> None:
        df_train = pd.read_json(
            os.path.join(self.global_config.working_dir, self.experiment.path_to_train_set),
            compression='gzip',
            lines=True
        )

        # df_val only contains the pair_ids for val-data. so we have to split manually and remove them as we do
        # not want to augment them
        df_val = pd.read_csv(os.path.join(self.global_config.working_dir, self.experiment.path_to_val_set))

        pair_ids_for_validation = df_val["pair_id"].values.tolist()
        df_train = df_train[~df_train["pair_id"].isin(pair_ids_for_validation)].reset_index(drop=True)

        df_train = set_datatypes_and_limit_string_length(df_train, self.experiment)

        # do augmentation on text
        if self.experiment.offline_augmentation is not None and len(self.experiment.offline_augmentation) > 0:
            columns_to_augment = [
                'title_left', 'description_left', 'brand_left', 'category_left',
                'title_right', 'description_right', 'brand_right', 'category_right'
            ]
            df_train = apply_offline_augmentation(
                df_train, columns_to_augment, self.experiment.offline_augmentation,
                experiment_configuration=self.experiment, global_configuration=self.global_config)

        df_train.to_parquet(os.path.join(self.output_path, f"custom_train_{self.filename}.parquet"))

        # save in ditto format
        write_wdc_data_in_ditto_format(df_train, os.path.join(self.output_path, f"ditto_train_{self.filename}"))

        self.write("")

    def output(self) -> luigi.LocalTarget:
        return self.make_output_target(self.output_path, f"prefix_{self.filename}_suffix")
