from data_aug_for_pm.utils.load_config import ExperimentConfiguration


def set_datatypes_and_limit_string_length(df, experiment: ExperimentConfiguration):
    str_cols = ['brand_left', 'title_left', 'category_left', 'description_left', 'price_left', 'brand_right',
                'title_right', 'category_right', 'description_right', 'price_right', "identifiers_left",
                "identifiers_right", "keyValuePairs_left", "keyValuePairs_right", "specTableContent_left",
                "specTableContent_right"]

    df[str_cols] = df[str_cols].astype(str)
    df["label"] = df["label"].astype(float)

    for col in str_cols:
        df[col] = df[col].apply(lambda x: _limit_string_length(x, experiment.max_string_len))

    return df


def _limit_string_length(string, max_length):
    if len(string) > max_length:
        return string[:max_length]
    else:
        return string
