import joblib
import luigi
import os
import hashlib

from data_aug_for_pm.tasks.base import LuigiBaseTask
from data_aug_for_pm.utils.load_config import load_global_config, load_config
from data_aug_for_pm.tasks.preprocess_mp_val_test_data import PreprocessMarktPilotValTestDataTask
from data_aug_for_pm.tasks.preprocess_mp_train_data import PreprocessMarktPilotTrainDataTask
from data_aug_for_pm.tasks.preprocess_wdc_val_test_data import PreprocessWdcValTestDataTask
from data_aug_for_pm.tasks.preprocess_wdc_train_data import PreprocessWDCTrainDataTask
from data_aug_for_pm.tasks.preprocess_magellan_train_data import PreprocessMagellanTrainDataTask
from data_aug_for_pm.tasks.preprocess_magellan_val_test_data import PreprocessMagellanValTestDataTask

# ditto imports
import sys
import data_aug_for_pm
sys.path.append(os.path.join(data_aug_for_pm.__path__[0], "matching_systems", "ditto"))
from data_aug_for_pm.matching_systems.ditto.train_ditto import main as ditto_main


class TrainDittoTask(LuigiBaseTask):
    experiment_name = luigi.Parameter()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.global_config = load_global_config()
        self.experiment = load_config(self.experiment_name)

        relevant_fields = self.experiment.__str__()
        self.filename = hashlib.md5(relevant_fields.encode()).hexdigest()
        self.output_path = os.path.join(self.global_config.working_dir, "matching_systems", "ditto")

    def requires(self):
        requirements = {}

        if self.experiment.dataset == "markt_pilot":
            requirements["train_data"] = PreprocessMarktPilotTrainDataTask(experiment_name=self.experiment_name)
            requirements["val_data"] = PreprocessMarktPilotValTestDataTask(experiment_name=self.experiment_name)
        elif self.experiment.dataset == "wdc":
            requirements["train_data"] = PreprocessWDCTrainDataTask(experiment_name=self.experiment_name)
            requirements["val_data"] = PreprocessWdcValTestDataTask(experiment_name=self.experiment_name)
        elif self.experiment.dataset in ["abt_buy", "amazon_google", "walmart_amazon"]:
            requirements["train_data"] = PreprocessMagellanTrainDataTask(experiment_name=self.experiment_name)
            requirements["val_data"] = PreprocessMagellanValTestDataTask(experiment_name=self.experiment_name)

        return requirements

    def run(self) -> None:
        train_input_path = self.input()["train_data"].path.replace("prefix_", "ditto_train_").replace("_suffix", ".txt")
        val_input_path = self.input()["val_data"].path.replace("prefix_", "ditto_val_").replace("_suffix", ".txt")
        test_input_path = self.input()["val_data"].path.replace("prefix_", "ditto_test_").replace("_suffix", ".txt")

        probabilities, epochs_trained = ditto_main(
            experiment=self.experiment,
            global_config=self.global_config,
            train_input_path=train_input_path,
            val_input_path=val_input_path,
            test_input_path=test_input_path
        )

        output = {"probabilities": probabilities, "epochs_trained": epochs_trained}
        joblib.dump(output, self.output().path)

    def output(self) -> luigi.LocalTarget:
        return self.make_output_target(self.output_path, f"{self.filename}.joblib")
