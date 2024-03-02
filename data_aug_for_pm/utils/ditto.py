import pandas as pd


def write_mp_data_in_ditto_format(df, output_path):
    cols_a = ["searched_brand", 'searched_number', 'searched_name', 'searched_description', 'searched_group']
    cols_b = ['found_brand', 'found_number', 'found_name', 'found_description', 'found_group']

    with open(f"{output_path}.txt", "w", encoding="utf-8", errors="replace") as f:
        for idx, row in df.iterrows():
            text = _get_row_text(row, cols_a, cols_b)
            f.write(text)


def write_wdc_data_in_ditto_format(df, output_path):
    cols_a = ['brand_left', 'title_left', 'category_left', 'description_left', 'price_left']
    cols_b = ['brand_right', 'title_right', 'category_right', 'description_right', 'price_right']

    with open(f"{output_path}.txt", "w", encoding="utf-8", errors="replace") as f:
        for idx, row in df.iterrows():
            text = _get_row_text(row, cols_a, cols_b)
            f.write(text)


def write_magellan_data_in_ditto_format(df: pd.DataFrame, output_path: str):
    cols = df.columns.tolist()
    cols_a = [i for i in cols if "left" in i]
    cols_b = [i for i in cols if "right" in i]

    with open(f"{output_path}.txt", "w", encoding="utf-8", errors="replace") as f:
        for idx, row in df.iterrows():
            text = _get_row_text(row, cols_a, cols_b)
            f.write(text)


def _clean_col_name(col_name: str):
    return (str(col_name)
            .replace("_searched", "")
            .replace("_found", "")
            .replace("_left", "")
            .replace("_right", "")
            .replace("left_", "")
            .replace("right_", ""))


def _get_row_text(row: pd.Series, cols_a: list, cols_b: list) -> str:
    text = ""
    for col in cols_a:
        text += "COL " + _clean_col_name(col) + " VAL " + str(row[col]).replace("\n", "").replace("\t", "") + " "
    text += "\t"

    for col in cols_b:
        text += "COL " + _clean_col_name(col) + " VAL " + str(row[col]).replace("\n", "").replace("\t", "") + " "
    text += "\t"

    text += str(row["label"]) + "\n"

    return text
