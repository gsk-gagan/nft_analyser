import json
import csv

from typing import Optional

import sqlite3
import pandas as pd


config = None
conn = None
    

def set_config(config_dict: dict):
    """Specify the configuration dict. The data sources are too big to include in the package. So please download them.
    Reference notebooks/main.ipynb notebook for the url of data sources to download them.

    Args:
        config_dict (dict): Contains local paths to data sources. Must contain paths for keys 'nft_sql' and 'glove_root'.
    """
    global config, conn
    config = config_dict
    # NFTS SQLite
    conn = sqlite3.connect(config['nft_sql'])


table_cache = {}
def get_table(table: str, count: Optional[int]=None, use_cached: bool=True) -> pd.DataFrame:
    """Obtains specified table from sqlite database.

    Args:
        table (str): Table name
        count (Optional[int], optional): The number of rows needed from the table. Defaults to None.
        use_cached (bool): If use table from the cache. We're passing results as reference to reduce memory load.

    Returns:
        pd.DataFrame: Table from the sqlite database in pandas.DataFrame format. Passed as reference to cached table.
    """
    if (table not in table_cache) or (not use_cached):
        query = f"SELECT * FROM {table}"
        df = pd.read_sql_query(query, conn)
        table_cache[table] = df
    return table_cache[table] if count is None else table_cache[table][:count]


glove_cache = {}
def get_glove(features: int=50) -> pd.DataFrame:
    """Obtains glove dataset of vector representation of words for the specified number of features.

    Args:
        features (int, optional): The size of the vector representing a given word. Defaults to 50.

    Returns:
        pd.DataFrame: Pre-trained glove dataset represented as pandas.DataFrame.
    """
    if features not in glove_cache:
        df = pd.read_csv(f"{config['glove_root']}/glove.6B.{features}d.txt",
                         sep=' ', index_col=0, header=None, quoting=csv.QUOTE_NONE)
        glove_cache[features] = df
    return glove_cache[features]
