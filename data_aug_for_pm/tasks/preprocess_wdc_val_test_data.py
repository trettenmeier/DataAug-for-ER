import luigi
import os
import pandas as pd
import hashlib

from data_aug_for_pm.tasks.base import LuigiBaseTask
from data_aug_for_pm.utils.load_config import load_global_config, load_config
from data_aug_for_pm.utils.wdc_dataset import set_datatypes_and_limit_string_length
from data_aug_for_pm.utils.ditto import write_wdc_data_in_ditto_format


class PreprocessWdcValTestDataTask(LuigiBaseTask):
    experiment_name = luigi.Parameter()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.global_config = load_global_config()
        self.experiment = load_config(self.experiment_name)

        # the relevant fields of the config get hashed, to detect unchanged fields in case of a rerun and save
        # computation time
        relevant_fields = " ".join([
            self.experiment.path_to_val_set,
            self.experiment.path_to_test_set
        ])
        self.filename = hashlib.md5(relevant_fields.encode()).hexdigest()
        self.output_path = os.path.join(self.global_config.working_dir, "data", "intermediate", "load_wdc_data")

    def run(self) -> None:
        df_train = pd.read_json(
            os.path.join(self.global_config.working_dir, self.experiment.path_to_train_set),
            compression='gzip',
            lines=True)

        # df_val only contains the pair_ids for val-data. so we have to split manually
        df_val = pd.read_csv(os.path.join(self.global_config.working_dir, self.experiment.path_to_val_set))

        df_test = pd.read_json(
            os.path.join(self.global_config.working_dir, self.experiment.path_to_test_set),
            compression='gzip',
            lines=True)

        pair_ids_for_validation = df_val["pair_id"].values.tolist()
        df_val = df_train[df_train["pair_id"].isin(pair_ids_for_validation)].reset_index(drop=True)

        # this is needed for the ditto-format
        df_val = df_val.replace(r'\t|\n', '', regex=True)
        df_test = df_test.replace(r'\t|\n', '', regex=True)

        df_val = set_datatypes_and_limit_string_length(df_val, self.experiment)
        df_test = set_datatypes_and_limit_string_length(df_test, self.experiment)

        df_val.to_parquet(os.path.join(self.output_path, f"custom_val_{self.filename}.parquet"))
        df_test.to_parquet(os.path.join(self.output_path, f"custom_test_{self.filename}.parquet"))

        # save in ditto format
        write_wdc_data_in_ditto_format(df_val, os.path.join(self.output_path, f"ditto_val_{self.filename}"))
        write_wdc_data_in_ditto_format(df_test, os.path.join(self.output_path, f"ditto_test_{self.filename}"))

        self.write("")

    def output(self) -> luigi.LocalTarget:
        return self.make_output_target(self.output_path, f"prefix_{self.filename}_suffix")
