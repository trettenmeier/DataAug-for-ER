import logging

logs = logging.getLogger()
logs.setLevel(logging.INFO)
logging.info("Logging initialized!")

import luigi

from data_aug_for_pm.tasks.run_experiment_task import RunExperimentTask


def main():
    luigi.build([RunExperimentTask(experiment_name="markt_pilot_m_bert_no_aug")], workers=1, local_scheduler=True)


if __name__ == "__main__":
    main()
