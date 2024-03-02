import mlflow

from data_aug_for_pm.utils.load_config import ExperimentConfiguration
from data_aug_for_pm.metrics.metricsbag import MetricsBag


def track_experiment(experiment: ExperimentConfiguration, bag: MetricsBag, epochs_trained=0):
    mlflow.end_run()

    with mlflow.start_run():
        
        mlflow.log_params({
            "name": experiment.name,
            "dataset": experiment.dataset,
            "subset size": get_subset_size(experiment.path_to_train_set),
            "path_to_train_set": experiment.path_to_train_set,
            "path_to_val_set": experiment.path_to_val_set,
            "path_to_test_set": experiment.path_to_test_set,
            "model": experiment.model,
            "offline_augmentation": experiment.offline_augmentation,
            "batch_size": experiment.batch_size,
            "max_string_len": experiment.max_string_len,
            "max_input_length": experiment.max_input_length,
            "epochs": experiment.epochs,
            "epochs_trained": epochs_trained
        })

        mlflow.log_metric("f1", round(bag.f1_scores[bag.index_of_maximum_f1_score], 3))
        mlflow.log_figure(bag.evaluate()[0], "metricsbag.png")


def get_subset_size(path):
    if "wdc" in path and "_xlarge" in path:
        return "xlarge"
    if "wdc" in path and "_large" in path:
        return "large"
    if "wdc" in path and "_medium" in path:
        return "medium"
    if "wdc" in path and "_small" in path:
        return "small"
    if "markt_pilot" in path and "l_" in path:
        return "large"
    if "markt_pilot" in path and "m_" in path:
        return "medium"
    if "markt_pilot" in path and "s_" in path:
        return "small"
    if "markt_pilot_dataset_train" in path:
        return "full"

    return "n.a."


def get_number_of_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
