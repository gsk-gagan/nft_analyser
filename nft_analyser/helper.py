import json
import csv
from importlib import resources
from typing import Optional

import sqlite3
import pandas as pd


with resources.open_text('data', "config.json") as f:
    config = json.load(f)
    
# NFTS SQLite
conn = sqlite3.connect(config['nft_sql'])

table_cache = {}
def get_table(table: str, count: Optional[int]=None):
    if table not in table_cache:
        query = f"SELECT * FROM {table}"
        df = pd.read_sql_query(query, conn)
        table_cache[table] = df
    return table_cache[table] if count is None else table_cache[table][:count]


# GLOVE
glove_cache = {}
def get_glove(features=50):
    if features not in glove_cache:
        df = pd.read_csv(f"{config['glove_root']}/glove.6B.{features}d.txt",
                         sep=' ', index_col=0, header=None, quoting=csv.QUOTE_NONE)
        glove_cache[features] = df
    return glove_cache[features]

    