import argparse
import os
import logging
import pytorch_lightning as pl
import pandas as pd
import torch
import nltk
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    AutoTokenizer,
    get_linear_schedule_with_warmup
)
from data_aug_for_pm.matching_systems.ditto.ditto_light.augment import Augmenter
from data_aug_for_pm.utils.load_config import ExperimentConfiguration, GlobalConfiguration

nltk.download('punkt')


class T5FineTuner(pl.LightningModule):
    def __init__(self, hparams, experiment_configuration: ExperimentConfiguration, global_configuration: GlobalConfiguration):
        super(T5FineTuner, self).__init__()
        self.hparams = hparams
        self.experiment_configuration = experiment_configuration
        self.global_configuration = global_configuration

        self.model = T5ForConditionalGeneration.from_pretrained(hparams.model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(hparams.tokenizer_name_or_path)

    def is_logger(self):
        return True

    def forward(
            self, input_ids, attention_mask=None, decoder_input_ids=None, decoder_attention_mask=None, lm_labels=None
    ):
        return self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=lm_labels,
        )

    def _step(self, batch):
        lm_labels = batch["target_ids"]
        lm_labels[lm_labels[:, :] == self.tokenizer.pad_token_id] = -100

        outputs = self(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            lm_labels=lm_labels,
            decoder_attention_mask=batch['target_mask']
        )

        loss = outputs[0]

        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)

        tensorboard_logs = {"train_loss": loss}
        return {"loss": loss, "log": tensorboard_logs}

    def training_epoch_end(self, outputs):
        avg_train_loss = torch.stack([x["loss"] for x in outputs]).mean()
        tensorboard_logs = {"avg_train_loss": avg_train_loss}
        return {"avg_train_loss": avg_train_loss, "log": tensorboard_logs, 'progress_bar': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch)
        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        tensorboard_logs = {"val_loss": avg_loss}
        return {"avg_val_loss": avg_loss, "log": tensorboard_logs, 'progress_bar': tensorboard_logs}

    def configure_optimizers(self):
        "Prepare optimizer and schedule (linear warmup and decay)"

        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)
        self.opt = optimizer
        return [optimizer]

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, second_order_closure=None):
        optimizer.step()
        optimizer.zero_grad()
        self.lr_scheduler.step()

    def get_tqdm_dict(self):
        tqdm_dict = {"loss": "{:.3f}".format(self.trainer.avg_loss), "lr": self.lr_scheduler.get_last_lr()[-1]}

        return tqdm_dict

    def train_dataloader(self):
        train_dataset = get_dataset(tokenizer=self.tokenizer, type_path=self.hparams.train_filename, args=self.hparams,
                                    experiment_configuration=self.experiment_configuration)
        dataloader = DataLoader(train_dataset, batch_size=self.hparams.train_batch_size, drop_last=True, shuffle=True,
                                num_workers=4)
        t_total = (
                (len(dataloader.dataset) // (self.hparams.train_batch_size * max(1, self.hparams.n_gpu)))
                // self.hparams.gradient_accumulation_steps
                * float(self.hparams.num_train_epochs)
        )
        scheduler = get_linear_schedule_with_warmup(
            self.opt, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=t_total
        )
        self.lr_scheduler = scheduler
        return dataloader

    def val_dataloader(self):
        val_dataset = get_dataset(tokenizer=self.tokenizer, type_path=self.hparams.valid_filename, args=self.hparams,
                                  experiment_configuration=self.experiment_configuration)
        return DataLoader(val_dataset, batch_size=self.hparams.eval_batch_size, num_workers=4)


logger = logging.getLogger(__name__)


class LoggingCallback(pl.Callback):
    def on_validation_end(self, trainer, pl_module):
        logger.info("***** Validation results *****")
        if pl_module.is_logger():
            metrics = trainer.callback_metrics
            # Log results
            for key in sorted(metrics):
                if key not in ["log", "progress_bar"]:
                    logger.info("{} = {}\n".format(key, str(metrics[key])))

    def on_test_end(self, trainer, pl_module):
        logger.info("***** Test results *****")

        if pl_module.is_logger():
            metrics = trainer.callback_metrics

            # Log and save results to file
            output_test_results_file = os.path.join(pl_module.hparams.output_dir, "test_results.txt")
            with open(output_test_results_file, "w") as writer:
                for key in sorted(metrics):
                    if key not in ["log", "progress_bar"]:
                        logger.info("{} = {}\n".format(key, str(metrics[key])))
                        writer.write("{} = {}\n".format(key, str(metrics[key])))


class ParaphraseDataset(Dataset):
    def __init__(self, tokenizer, data_dir, type_path, max_len, experiment_configuration: ExperimentConfiguration):
        self.path = os.path.join(data_dir, type_path)
        self.experiment_configuration = experiment_configuration
        self.data_dir = data_dir

        # orig: data is already in ditto format: "left_string \t right_string \t label"...
        # self.data = pd.read_csv(self.path, sep='\t', header=None)
        # load data and convert to ditto format
        self.data = preprocess_data(self.data_dir, type_path, self.experiment_configuration)

        # ...so the column 0 is the left entity (all fields of it)
        self.source_column = self.target_column = 0
        if len(self.data.keys()) == 3 and str(self.data[1][0]) != 'nan':
            self.target_column = 1

        self.max_len = max_len
        self.tokenizer = tokenizer
        self.inputs = []
        self.targets = []

        self._build()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        source_ids = self.inputs[index]["input_ids"]  # .squeeze()
        target_ids = self.targets[index]["input_ids"]  # .squeeze()

        src_mask = self.inputs[index]["attention_mask"]  # .squeeze()  # might need to squeeze
        target_mask = self.targets[index]["attention_mask"]  # .squeeze()  # might need to squeeze

        return {"source_ids": torch.LongTensor(source_ids),
                "source_mask": torch.LongTensor(src_mask),
                "target_ids": torch.LongTensor(target_ids),
                "target_mask": torch.LongTensor(target_mask)}

    def _build(self):
        ag = Augmenter()
        for idx in tqdm(range(len(self.data))):
            left = self.data.loc[idx, self.source_column]
            right = self.data.loc[idx, self.target_column]

            if left != right:
                original = left + ' [SEP] ' + right
            else:
                original = left
            corrupted = ag.augment_sent(original, op='corrupt')
            input_ = ("corrupt: " + corrupted + " </s>").replace('[SEP]', '***')  # avoid the special token
            target = (original + " </s>").replace('[SEP]', '***')

            if idx == 0:
                print(input_, target)

            # tokenize inputs
            tokenized_inputs = self.tokenizer.encode_plus(
                input_, max_length=self.max_len, pad_to_max_length=True, truncation=True)
            # tokenize targets
            tokenized_targets = self.tokenizer.encode_plus(
                target, max_length=self.max_len, pad_to_max_length=True, truncation=True
            )

            self.inputs.append(tokenized_inputs)
            self.targets.append(tokenized_targets)


def get_dataset(tokenizer, type_path, args, experiment_configuration: ExperimentConfiguration):
    # type_path is train-file or valid-file
    return ParaphraseDataset(tokenizer=tokenizer,
                             data_dir=args.data_dir,
                             type_path=type_path,
                             max_len=experiment_configuration.max_input_length,
                             experiment_configuration=experiment_configuration)


def preprocess_data(data_dir: str, type_path: str, experiment_configuration: ExperimentConfiguration) -> pd.DataFrame:
    if "amazon_google" in experiment_configuration.dataset:
        from data_aug_for_pm.utils.magellan import amazon_google_to_dataframe

        if "train" in type_path:
            df = amazon_google_to_dataframe(os.path.join(data_dir, experiment_configuration.path_to_train_set))
        else:
            df = amazon_google_to_dataframe(os.path.join(data_dir, experiment_configuration.path_to_val_set))

        cols = df.columns
        cols_a = [i for i in cols if "left" in i]
        cols_b = [i for i in cols if "right" in i]

        return create_df(df, cols_a, cols_b)
    elif "walmart_amazon" in experiment_configuration.dataset:
        from data_aug_for_pm.utils.magellan import walmart_amazon_to_dataframe

        if "train" in type_path:
            df = walmart_amazon_to_dataframe(os.path.join(data_dir, experiment_configuration.path_to_train_set))
        else:
            df = walmart_amazon_to_dataframe(os.path.join(data_dir, experiment_configuration.path_to_val_set))

        cols = df.columns
        cols_a = [i for i in cols if "left" in i]
        cols_b = [i for i in cols if "right" in i]

        return create_df(df, cols_a, cols_b)
    elif "markt_pilot" in experiment_configuration.dataset:
        if "train" in type_path:
            df = pd.read_parquet(os.path.join(data_dir, experiment_configuration.path_to_train_set))
        else:
            df = pd.read_parquet(os.path.join(data_dir, experiment_configuration.path_to_val_set))

        cols_a = ["searched_brand", 'searched_number', 'searched_name', 'searched_description', 'searched_group']
        cols_b = ['found_brand', 'found_number', 'found_name', 'found_description', 'found_group']

        return create_df(df, cols_a, cols_b)
    elif "wdc" in experiment_configuration.dataset:
        if "train" in type_path:
            df = pd.read_json(os.path.join(data_dir, experiment_configuration.path_to_train_set),
                              compression='gzip',
                              lines=True)
        else:
            df_train = pd.read_json(os.path.join(data_dir, experiment_configuration.path_to_train_set),
                                    compression='gzip',
                                    lines=True)
            df_val = pd.read_csv(os.path.join(data_dir, experiment_configuration.path_to_val_set))

            pair_ids_for_validation = df_val["pair_id"].values.tolist()
            df = df_train[df_train["pair_id"].isin(pair_ids_for_validation)].reset_index(drop=True)

        cols_a = ['brand_left', 'title_left', 'category_left', 'description_left', 'price_left']
        cols_b = ['brand_right', 'title_right', 'category_right', 'description_right', 'price_right']

        return create_df(df, cols_a, cols_b)
    else:
        raise NotImplementedError()


def create_df(df, cols_a, cols_b):
    df[cols_a] = df[cols_a].astype(str)
    df[cols_b] = df[cols_b].astype(str)

    values_for_df_new = {
        0: [],
        1: [],
        2: []
    }
    for _, row in df.iterrows():
        values_for_df_new[0].append("".join(row[cols_a].values.tolist()))
        values_for_df_new[1].append("".join(row[cols_b].values.tolist()))
        values_for_df_new[2].append(row["label"])

    return pd.DataFrame(values_for_df_new)


def main(experiment_configuration: ExperimentConfiguration, global_configuration: GlobalConfiguration):
    model_output_dir = os.path.join(global_configuration.working_dir, "invda_models")
    os.makedirs(model_output_dir, exist_ok=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_len", type=int, default=experiment_configuration.max_input_length)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--n_epochs", type=int, default=3)
    parser.add_argument("--logdir", type=str, default="checkpoints/")
    parser.add_argument("--model_name_or_path", type=str, default="t5-base")
    parser.add_argument("--tokenizer_name_or_path", type=str, default="t5-base")
    parser.add_argument("--fp16", dest="fp16", action="store_true")
    parser.add_argument("--data_dir", type=str, default=global_configuration.working_dir)
    parser.add_argument("--train_filename", type=str, default='log_no_dbpedia_train_10ptg')
    parser.add_argument("--valid_filename", type=str, default='log_no_dbpedia_valid_10ptg')
    parser.add_argument("--model_output_dir", type=str, default=model_output_dir)
    parser.add_argument("--gpu_list", type=str, default='0')
    parser.add_argument("--type", type=str, default='em')  # two mode: cls or em (cleaning)

    hp = parser.parse_args()

    args_dict = dict(
        data_dir=hp.data_dir,  # path for data files
        output_dir=hp.model_output_dir,  # path to save the checkpoints
        model_name_or_path=hp.model_name_or_path,
        tokenizer_name_or_path=hp.tokenizer_name_or_path,
        max_seq_length=hp.max_len,
        learning_rate=hp.lr,
        weight_decay=0.0,
        adam_epsilon=1e-8,
        warmup_steps=0,
        train_batch_size=hp.batch_size,
        eval_batch_size=hp.batch_size,
        gradient_accumulation_steps=16,
        gpus=hp.gpu_list,
        n_gpu=len(hp.gpu_list.split(',')),
        early_stop_callback=False,
        fp_16=False,
        opt_level='O1',
        # you can find out more on optimisation levels here https://nvidia.github.io/apex/amp.html#opt-levels-and-properties
        max_grad_norm=1.0,  # if you enable 16-bit training then set this to a sensible value, 0.5 is a good default
        seed=42,
        train_filename=hp.train_filename,
        valid_filename=hp.valid_filename
    )

    tokenizer = AutoTokenizer.from_pretrained(hp.tokenizer_name_or_path)

    dataset = ParaphraseDataset(tokenizer, hp.data_dir, hp.valid_filename, experiment_configuration.max_input_length,
                                experiment_configuration=experiment_configuration)
    print("Val dataset: ", len(dataset))

    data = dataset[61]
    print(tokenizer.decode(data['source_ids']))
    print(tokenizer.decode(data['target_ids']))

    if not os.path.exists(hp.model_output_dir):
        os.makedirs(hp.model_output_dir)

    args_dict.update({
        'data_dir': hp.data_dir,
        'output_dir': hp.model_output_dir,
        'num_train_epochs': hp.n_epochs,
        'max_seq_length': hp.max_len})
    args = argparse.Namespace(**args_dict)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        filepath=args.output_dir, prefix="checkpoint", monitor="val_loss", mode="min", save_top_k=1
    )

    train_params = dict(
        accumulate_grad_batches=args.gradient_accumulation_steps,
        gpus=args.gpus,
        max_epochs=args.num_train_epochs,
        early_stop_callback=False,
        precision=16 if args.fp_16 else 32,
        amp_level=args.opt_level,
        gradient_clip_val=args.max_grad_norm,
        checkpoint_callback=False,
        callbacks=[LoggingCallback()],
    )

    print("Initialize model")
    model = T5FineTuner(args, experiment_configuration=experiment_configuration, global_configuration=global_configuration)

    trainer = pl.Trainer(**train_params)

    print("Training model")
    trainer.fit(model)
    print("Training finished")

    print("Saving model")
    model.model.save_pretrained(hp.model_output_dir)

    print("Saved model")
    return model.model, tokenizer
