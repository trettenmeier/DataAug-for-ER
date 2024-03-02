from data_aug_for_pm.utils.load_config import ExperimentConfiguration


def set_datatypes_and_limit_string_length(df, experiment: ExperimentConfiguration):
    str_cols = ['searched_brand', 'searched_number', 'searched_name', 'searched_description', 'searched_group',
                'found_brand', 'found_number', 'found_name', 'found_description', 'found_group']

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
