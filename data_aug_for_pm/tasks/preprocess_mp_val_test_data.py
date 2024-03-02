import luigi
import os
import pandas as pd
import hashlib

from data_aug_for_pm.tasks.base import LuigiBaseTask
from data_aug_for_pm.utils.load_config import load_global_config, load_config
from data_aug_for_pm.utils.mp_dataset import set_datatypes_and_limit_string_length
from data_aug_for_pm.utils.ditto import write_mp_data_in_ditto_format


class PreprocessMarktPilotValTestDataTask(LuigiBaseTask):
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
        self.output_path = os.path.join(self.global_config.working_dir, "data", "intermediate", "load_mp_data")

    def run(self) -> None:
        df_val = pd.read_parquet(os.path.join(self.global_config.working_dir, self.experiment.path_to_val_set))
        df_test = pd.read_parquet(os.path.join(self.global_config.working_dir, self.experiment.path_to_test_set))

        df_val = set_datatypes_and_limit_string_length(df_val, self.experiment)
        df_test = set_datatypes_and_limit_string_length(df_test, self.experiment)

        df_val.to_parquet(os.path.join(self.output_path, f"custom_val_{self.filename}.parquet"))
        df_test.to_parquet(os.path.join(self.output_path, f"custom_test_{self.filename}.parquet"))

        # save in ditto format
        write_mp_data_in_ditto_format(df_val, os.path.join(self.output_path, f"ditto_val_{self.filename}"))
        write_mp_data_in_ditto_format(df_test, os.path.join(self.output_path, f"ditto_test_{self.filename}"))

        self.write("")

    def output(self) -> luigi.LocalTarget:
        return self.make_output_target(self.output_path, f"prefix_{self.filename}_suffix")
