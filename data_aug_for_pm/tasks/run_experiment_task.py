import luigi
import os
import hashlib
import logging
import pandas as pd
import joblib
import time

from data_aug_for_pm.tasks.base import LuigiBaseTask
from data_aug_for_pm.dataloader.mp_non_bert_loader import MarktPilotNonBertLoader
from data_aug_for_pm.dataloader.wdc_non_bert_loader import WdcNonBertLoader
from data_aug_for_pm.dataloader.mp_bert_loader import MarktPilotBertLoader
from data_aug_for_pm.dataloader.wdc_bert_loader import WdcBertLoader
from data_aug_for_pm.dataloader.magellan_bert_loader import MagellanBertLoader
from data_aug_for_pm.dataloader.magellan_non_bert_loader import MagellanNonBertLoader
from data_aug_for_pm.utils.load_config import load_global_config, load_config
from data_aug_for_pm.tasks.preprocess_mp_val_test_data import PreprocessMarktPilotValTestDataTask
from data_aug_for_pm.tasks.preprocess_mp_train_data import PreprocessMarktPilotTrainDataTask
from data_aug_for_pm.tasks.preprocess_wdc_val_test_data import PreprocessWdcValTestDataTask
from data_aug_for_pm.tasks.preprocess_wdc_train_data import PreprocessWDCTrainDataTask
from data_aug_for_pm.tasks.preprocess_magellan_train_data import PreprocessMagellanTrainDataTask
from data_aug_for_pm.tasks.preprocess_magellan_val_test_data import PreprocessMagellanValTestDataTask
from data_aug_for_pm.trainer.bert_trainer import Trainer as BertTrainer
from data_aug_for_pm.trainer.non_bert_trainer import Trainer as NonBertTrainer
from data_aug_for_pm.metrics.metricsbag import MetricsBag
from data_aug_for_pm.utils.mlflow_tracking import track_experiment
from data_aug_for_pm.models.bert import get_model as get_bert_model
from data_aug_for_pm.utils.mlflow_tracking import get_number_of_trainable_params
from data_aug_for_pm.tasks.train_ditto import TrainDittoTask


class RunExperimentTask(LuigiBaseTask):
    experiment_name = luigi.Parameter()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.global_config = load_global_config()
        self.experiment = load_config(self.experiment_name)

        relevant_fields = self.experiment.__str__()
        self.filename = hashlib.md5(relevant_fields.encode()).hexdigest()
        self.output_path = os.path.join(self.global_config.working_dir, "experiment_results")

    def requires(self):
        requirements = {}

        if self.experiment.model == "ditto":
            requirements["ditto_results"] = TrainDittoTask(experiment_name=self.experiment_name)

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
        start_time = time.time()

        if self.experiment.model == "ditto":
            ditto_output = joblib.load(self.input()["ditto_results"].path)
            probabilities = ditto_output["probabilities"]
            epochs_trained = ditto_output["epochs_trained"]
        else:
            probabilities, epochs_trained = self.run_task()

        test_input_path = self.input()["val_data"].path.replace("prefix_", "custom_test_").replace("_suffix", ".parquet")
        df_test = pd.read_parquet(test_input_path)
        true_labels = df_test.label.to_list()

        try:
            bag = MetricsBag(y=true_labels, y_hat=probabilities[:, 1])
        except:
            bag = MetricsBag(y=true_labels, y_hat=probabilities)
        bag.evaluate()

        end_time = time.time()
        duration_in_sec = end_time - start_time
        duration_in_min = round(duration_in_sec / 60)
        duration_string = f"Execution took {duration_in_min} minutes"

        info_string = f"Max. F1-score: {round(bag.f1_scores[bag.index_of_maximum_f1_score], 4)}"
        logging.info(info_string)
        logging.info(duration_string)

        track_experiment(self.experiment, bag, epochs_trained)

        self.write("\n".join([
            info_string,
            self.experiment.print_data(),
            f"Epochs actually trained: {epochs_trained}",
            duration_string
        ]))

    def output(self) -> luigi.LocalTarget:
        return self.make_output_target(self.output_path, f"{self.experiment.name}_{self.filename}.txt")

    def run_task(self):
        # get data
        train_input_path = self.input()["train_data"].path.replace("prefix_", "custom_train_").replace("_suffix", ".parquet")
        df_train = pd.read_parquet(train_input_path)

        val_input_path = self.input()["val_data"].path.replace("prefix_", "custom_val_").replace("_suffix", ".parquet")
        df_val = pd.read_parquet(val_input_path)

        test_input_path = self.input()["val_data"].path.replace("prefix_", "custom_test_").replace("_suffix", ".parquet")
        df_test = pd.read_parquet(test_input_path)

        if self.experiment.dataset == "markt_pilot":
            if self.experiment.model == "bert":
                loader_factory = MarktPilotBertLoader(df_train, df_val, df_test, self.experiment)
            else:
                loader_factory = MarktPilotNonBertLoader(df_train, df_val, df_test, self.experiment)

        elif self.experiment.dataset == "wdc":
            if self.experiment.model == "bert":
                loader_factory = WdcBertLoader(df_train, df_val, df_test, self.experiment)
            else:
                loader_factory = WdcNonBertLoader(df_train, df_val, df_test, self.experiment)

        elif self.experiment.dataset in ["abt_buy", "amazon_google", "walmart_amazon"]:
            if self.experiment.model == "bert":
                loader_factory = MagellanBertLoader(df_train, df_val, df_test, self.experiment)
            else:
                loader_factory = MagellanNonBertLoader(df_train, df_val, df_test, self.experiment)

        else:
            raise ValueError("Unknown dataset name.")

        train_loader = loader_factory.get_train_loader()
        val_loader = loader_factory.get_val_loader()
        test_loader = loader_factory.get_test_loader()

        # get model
        if self.experiment.model == "bert":
            model = get_bert_model(self.experiment)
        else:
            raise ValueError("Unknown model name.")

        logging.info(f"Number of trainable model parameters: {get_number_of_trainable_params(model):,}")

        # get trainer
        if self.experiment.model == "bert":
            trainer = BertTrainer(
                model=model,
                train_dataloader=train_loader,
                val_dataloader=val_loader,
                experiment=self.experiment,
                working_dir=self.global_config.working_dir
            )
        else:
            raise ValueError("Unkown model name when trying to select trainer.)")

        # run
        trainer.train()

        epochs_trained = trainer.training_stats[-1]['epoch']
        probabilities, _ = trainer.evaluate(test_loader)

        return probabilities, epochs_trained
