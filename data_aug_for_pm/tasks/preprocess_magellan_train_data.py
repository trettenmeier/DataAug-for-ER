import luigi
import os
import hashlib

from data_aug_for_pm.tasks.base import LuigiBaseTask
from data_aug_for_pm.utils.load_config import load_global_config, load_config
from data_aug_for_pm.utils.ditto import write_magellan_data_in_ditto_format
from data_aug_for_pm.utils.magellan import abt_buy_to_dataframe, amazon_google_to_dataframe, walmart_amazon_to_dataframe
from data_aug_for_pm.augmenter.offline_augmentation import apply_offline_augmentation


class PreprocessMagellanTrainDataTask(LuigiBaseTask):
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
        self.output_path = os.path.join(self.global_config.working_dir, "data", "intermediate", self.experiment.dataset)

    def run(self) -> None:
        if self.experiment.dataset == "abt_buy":
            df_train = abt_buy_to_dataframe(os.path.join(self.global_config.working_dir, self.experiment.path_to_train_set))
        elif self.experiment.dataset == "amazon_google":
            df_train = amazon_google_to_dataframe(os.path.join(self.global_config.working_dir, self.experiment.path_to_train_set))
        elif self.experiment.dataset == "walmart_amazon":
            df_train = walmart_amazon_to_dataframe(
                os.path.join(self.global_config.working_dir, self.experiment.path_to_train_set))
        else:
            raise ValueError

        # do augmentation on text
        if self.experiment.offline_augmentation is not None and len(self.experiment.offline_augmentation) > 0:
            columns_in_df = df_train.columns.tolist()
            columns_to_augment = [i for i in columns_in_df if ("left" in i or "right" in i) and "price" not in i]
            df_train = apply_offline_augmentation(
                df_train, columns_to_augment, self.experiment.offline_augmentation,
                experiment_configuration=self.experiment, global_configuration=self.global_config)

        df_train.to_parquet(os.path.join(self.output_path, f"custom_train_{self.filename}.parquet"))

        # save in ditto format
        write_magellan_data_in_ditto_format(df_train, os.path.join(self.output_path, f"ditto_train_{self.filename}"))

        self.write("")

    def output(self) -> luigi.LocalTarget:
        return self.make_output_target(self.output_path, f"prefix_{self.filename}_suffix")
