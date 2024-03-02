import os
import argparse
import json
import sys
from typing import Optional

import torch
import numpy as np
import random
import hashlib

sys.path.insert(0, "Snippext_public")

from ditto_light.dataset import DittoDataset
from ditto_light.summarize import Summarizer
from ditto_light.knowledge import *
from ditto_light.ditto import train

from data_aug_for_pm.utils.load_config import ExperimentConfiguration, GlobalConfiguration
from dataclasses import dataclass


def main(experiment: ExperimentConfiguration, global_config: GlobalConfiguration, train_input_path, val_input_path,
         test_input_path):
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--task", type=str, default="Structured/Beer")
    # parser.add_argument("--run_id", type=int, default=0)
    # parser.add_argument("--batch_size", type=int, default=64)
    # parser.add_argument("--max_len", type=int, default=256)
    # parser.add_argument("--lr", type=float, default=3e-5)
    # parser.add_argument("--n_epochs", type=int, default=20)
    # parser.add_argument("--finetuning", dest="finetuning", action="store_true")
    # parser.add_argument("--save_model", dest="save_model", action="store_true")
    # parser.add_argument("--logdir", type=str, default="checkpoints/")
    # parser.add_argument("--lm", type=str, default='distilbert')
    # parser.add_argument("--fp16", dest="fp16", action="store_true")
    # parser.add_argument("--da", type=str, default=None)
    # parser.add_argument("--alpha_aug", type=float, default=0.8)
    # parser.add_argument("--dk", type=str, default=None)
    # parser.add_argument("--summarize", dest="summarize", action="store_true")
    # parser.add_argument("--size", type=int, default=None)

    @dataclass
    class Configuration:
        task: str
        lm: str
        size: Optional[int]
        max_len: int
        da: Optional[str]
        dk: Optional[list]
        summarize: bool
        lr: float
        n_epochs: int
        finetuning: bool
        save_model: bool
        logdir: str
        fp16: bool
        alpha_aug: float
        run_id: int
        batch_size: int
        train_input_path: str
        val_input_path: str
        test_input_path: str

    hp = Configuration(
        task=experiment.name,
        lm="distilbert",
        size=None,
        max_len=experiment.max_input_length,
        da="all" if "mixda" in experiment.online_augmentation else None,  # todo: test!
        dk=None,
        summarize=True,
        lr=3e-5,
        n_epochs=experiment.epochs,
        finetuning=True,
        save_model=True,
        logdir=os.path.join(global_config.working_dir, "ditto_logdir"),
        fp16=False,
        alpha_aug=0.8,
        batch_size=experiment.batch_size,
        run_id=0,
        train_input_path=train_input_path,
        val_input_path=val_input_path,
        test_input_path=test_input_path
    )

    # hash of config
    config_hash = experiment.__str__()
    config_hash = hashlib.md5(config_hash.encode()).hexdigest()

    # set seeds
    # seed = hp.run_id
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # only a single task for baseline
    # task = hp.task

    # create the tag of the run
    run_tag = '%s_lm=%s_hash=%s_run_id=%s' % (hp.task, hp.lm, config_hash, str(random.randint(0, 999999)))
    run_tag = run_tag.replace('/', '_')

    # load task configuration
    # configs = json.load(open('configs.json'))
    # configs = {conf['name']: conf for conf in configs}
    # config = configs[task]

    trainset = hp.train_input_path
    validset = hp.val_input_path
    testset = hp.test_input_path

    # summarize the sequences up to the max sequence length
    if hp.summarize:
        # summarizer = Summarizer(config, lm=lm)
        summarizer = Summarizer(hp, lm=hp.lm)
        trainset = summarizer.transform_file(trainset, max_len=hp.max_len, overwrite=True)
        validset = summarizer.transform_file(validset, max_len=hp.max_len, overwrite=True)
        testset = summarizer.transform_file(testset, max_len=hp.max_len, overwrite=True)

    if hp.dk is not None:
        if hp.dk == 'product':
            injector = ProductDKInjector(experiment=experiment, global_config=global_config, name=hp.dk)
        else:
            injector = GeneralDKInjector(experiment=experiment, global_config=global_config, name=hp.dk)

        trainset = injector.transform_file(trainset)
        validset = injector.transform_file(validset)
        testset = injector.transform_file(testset)

    # load train/dev/test sets
    train_dataset = DittoDataset(trainset,
                                 lm=hp.lm,
                                 max_len=hp.max_len,
                                 size=hp.size,
                                 da=hp.da)
    valid_dataset = DittoDataset(validset, lm=hp.lm)
    test_dataset = DittoDataset(testset, lm=hp.lm)

    # train and evaluate the model
    all_probs, epochs_trained = train(train_dataset, valid_dataset, test_dataset, run_tag, hp, experiment)
    return all_probs, epochs_trained

# if __name__=="__main__":
#     main()
