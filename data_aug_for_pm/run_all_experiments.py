import logging
import torch
import multiprocessing
import platform
import luigi
import os

import data_aug_for_pm.experiments
from data_aug_for_pm.tasks.base import LuigiBaseTask
from data_aug_for_pm.utils.load_config import load_global_config
from data_aug_for_pm.tasks.run_experiment_task import RunExperimentTask

logs = logging.getLogger()
logs.setLevel(logging.INFO)
logging.info("Logging initialized!")


class RunAllExperimentsTask(LuigiBaseTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.global_config = load_global_config()
        self.output_path = os.path.join(self.global_config.working_dir, "experiment_results")

    def requires(self):
        directory_path = os.path.abspath(data_aug_for_pm.experiments.__file__).replace("__init__.py", "")
        experiments = RunAllExperimentsTask.get_filenames_without_extension(directory_path)
        return [RunExperimentTask(experiment_name=i) for i in experiments]

    def run(self) -> None:
        pass

    def output(self) -> luigi.LocalTarget:
        return self.make_output_target(self.output_path, "dummy.txt")

    @staticmethod
    def get_filenames_without_extension(directory_path):
        filenames = []
        for filename in os.listdir(directory_path):
            name_without_extension = os.path.splitext(filename)[0]
            filenames.append(name_without_extension)
        return [i for i in filenames if ("__init__" not in i and "__pycache__" not in i)]


def main():
    # make sure to start 'luigid' beforehand (or set local_scheduler to True)
    assert torch.cuda.is_available(), "No CUDA found. Aborting."
    luigi.build([RunAllExperimentsTask()], local_scheduler=False, workers=1)


if __name__ == '__main__':
    if platform.system() == 'Windows':
        multiprocessing.freeze_support()

    main()
    os.system("vastai stop instance $CONTAINER_ID;")

