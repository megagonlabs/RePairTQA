import os
import sys
import json
from collections import defaultdict
import pandas as pd


def load_all_templates(paths):
    all_templates = defaultdict()
    for path in paths:
        with open(path, "r") as f:
            templates = json.load(f)
            all_templates[path.split("/")[-1].split(".")[0]] = templates
    return all_templates

def read_json(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data

def table_dict_to_df(table_dict):
    columns = table_dict["table_columns"]
    content = table_dict["table_content"]
    return pd.DataFrame(content, columns=columns)

def df_to_dict_table(df):
    """
    Turn DataFrame to {'table_columns': [...], 'table_content': [[...], ...]} format
    """
    return {
        "table_columns": list(df.columns),
        "table_content": df.values.tolist()
    }

def flatten_list(lst):
    for item in lst:
        if isinstance(item, list):
            yield from flatten_list(item)
        else:
            yield item

def normalize(x):
    if isinstance(x, list):
        flat = list(flatten_list(x))
        return " ".join(map(str, flat)).lower()
    elif isinstance(x, dict):
        return json.dumps(x, sort_keys=True).lower()
    elif isinstance(x, str):
        return x.lower()
    else:
        return str(x).lower()

def eval_em_pm(pred, gt):
    p = normalize(pred)
    g = normalize(gt)
    em = int(p == g)
    pm = int(p in g or g in p)
    return em, pm