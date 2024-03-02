import os
import pandas as pd


def abt_buy_to_dataframe(path_to_csv: str) -> pd.DataFrame:
    df = pd.read_csv(path_to_csv)

    left_table = pd.read_csv(os.path.join(os.path.dirname(path_to_csv), "tableA.csv"))
    right_table = pd.read_csv(os.path.join(os.path.dirname(path_to_csv), "tableB.csv"))

    df = pd.merge(left=df, right=left_table, left_on="ltable_id", right_on="id", how="left")
    df = df.rename(columns={
        "name": "left_name",
        "description": "left_description",
        "price": "left_price"
    }).drop(columns=["id"])

    df = pd.merge(left=df, right=right_table, left_on="rtable_id", right_on="id", how="left")
    df = df.rename(columns={
        "name": "right_name",
        "description": "right_description",
        "price": "right_price"
    }).drop(columns=["id"])

    return df


def amazon_google_to_dataframe(path_to_csv: str) -> pd.DataFrame:
    df = pd.read_csv(path_to_csv)

    left_table = pd.read_csv(os.path.join(os.path.dirname(path_to_csv), "tableA.csv"))
    right_table = pd.read_csv(os.path.join(os.path.dirname(path_to_csv), "tableB.csv"))

    df = pd.merge(left=df, right=left_table, left_on="ltable_id", right_on="id", how="left")
    df = df.rename(columns={
        "title": "left_title",
        "manufacturer": "left_manufacturer",
        "price": "left_price"
    }).drop(columns=["id"])

    df = pd.merge(left=df, right=right_table, left_on="rtable_id", right_on="id", how="left")
    df = df.rename(columns={
        "title": "right_title",
        "manufacturer": "right_manufacturer",
        "price": "right_price"
    }).drop(columns=["id"])

    return df


def walmart_amazon_to_dataframe(path_to_csv: str) -> pd.DataFrame:
    df = pd.read_csv(path_to_csv)

    left_table = pd.read_csv(os.path.join(os.path.dirname(path_to_csv), "tableA.csv"))
    right_table = pd.read_csv(os.path.join(os.path.dirname(path_to_csv), "tableB.csv"))

    df = pd.merge(left=df, right=left_table, left_on="ltable_id", right_on="id", how="left")
    df = df.rename(columns={
        "title": "left_title",
        "category": "left_category",
        "brand": "left_brand",
        "modelno": "left_modelno",
        "price": "left_price"
    }).drop(columns=["id"])

    df = pd.merge(left=df, right=right_table, left_on="rtable_id", right_on="id", how="left")
    df = df.rename(columns={
        "title": "right_title",
        "category": "right_category",
        "brand": "right_brand",
        "modelno": "right_modelno",
        "price": "right_price"
    }).drop(columns=["id"])

    return df
