import pandas as pd

from data_aug_for_pm.dataloader.mp_non_bert_loader import MarktPilotNonBertLoader


class MagellanNonBertLoader(MarktPilotNonBertLoader):
    def _get_sentences_and_labels(self, df: pd.DataFrame):
        cols = df.columns.tolist()
        cols_left = [i for i in cols if "left" in i]
        cols_right = [i for i in cols if "right" in i]

        df = df.fillna("")

        def page_concat(x):
            return " ".join([str(x[i]) for i in cols_left])

        def client_concat(x):
            return " ".join([str(x[i]) for i in cols_right])

        df["page_concat"] = df.apply(page_concat, axis=1)
        df["client_concat"] = df.apply(client_concat, axis=1)

        # cap length
        def cap_len_page(x):
            return x["page_concat"][:self.experiment.max_string_len]

        def cap_len_client(x):
            return x["client_concat"][:self.experiment.max_string_len]

        df["page_concat"] = df.apply(cap_len_page, axis=1)
        df["client_concat"] = df.apply(cap_len_client, axis=1)

        # get lists of sentences and labels
        page_sentences = df["page_concat"].values
        client_sentences = df["client_concat"].values
        labels = df.label.astype(float).values

        return client_sentences, page_sentences, labels
