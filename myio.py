#!/usr/bin/env python3

import os, random, sys, hashlib
import numpy as np
import pandas as pd
from analytics.bamboo import Bamboo as bb


# HTML to restart kernel in jupyter notebook
# Usage: from IPython.core.display import HTML; HTML(RESTART_KERNEL)
RESTART_KERNEL = "<script>Jupyter.notebook.kernel.restart()</script>"


class QueryIO(object):

    GLOBAL_QUERY_CACHE = {}

    def query_with_cache(self, namespace, sql, issave=True, file_name=None):
        key = '{namespace}.{sql}'.format(**locals())
        file_name = file_name or hashlib.sha1(str.encode(key)).hexdigest() + '.csv'
        if key in QueryIO.GLOBAL_QUERY_CACHE:
            data = QueryIO.GLOBAL_QUERY_CACHE[key]
            print('Cached data loaded')
        elif os.path.exists(file_name):
            data = pd.read_csv(file_name)
            print('Local data loaded')
        else:  # no cache or file
            print('Querying data')
            data = bb.query_presto(namespace, sql)
            if issave:
                data.to_csv(file_name, index=False)
                print('Saved as ' + file_name)
                with open("SavedFileIndex.txt", "a") as myfile:
                    myfile.write("{}: \n{}\n\n".format(file_name, key))
        QueryIO.GLOBAL_QUERY_CACHE[key] = data.copy()
        return data


class DataIO(object):
    """Read data as pandas dataframe from local file or query """

    def __init__(self):
        self.dir_name = ""
        self.file_name = ""
        self.query = None  # A dict of 'db' and 'query'
        self.df = None
        self.load_source = ""

    def _set_load_source(self):
        if self.load_source not in ("auto", "file", "query"):
            print("Wrong load source")
            sys.exit(1)
        if self.load_source == "auto":
            self.load_source = "file" if os.path.exists(self.file_name) else "query"

    def _get_query(self, query, query_file):
        if query is not None:
            self.query = query
        elif query_file is not None:
            with open(query_file) as f:
                query_text = f.read()
            self.query = {"query": query_text, "db": "hive"}

    def _load_data(self):
        if self.load_source == "file":
            self.df = pd.read_csv(self.file_name)
            print("Local data loaded")
        else:
            print("Querying data... ")
            if self.query is None:
                print("No query provided")
                sys.exit(1)
            # data = QR.execute(self.query['db'], self.query['query'],
            # timeout=10**6).fetchall()
            # self.df = pd.DataFrame(data)

    def _save_data(self):
        self.df.to_csv(self.file_name, index=False)
        print("Data saved to {}".format(self.file_name))

    def _set_time_col(self, col_time):
        if col_time is not None and self.df[col_time].dtype.kind != "M":
            self.df[col_time] = pd.to_datetime(self.df[col_time])

    def set_file_name(self, data_file, dir_name=""):
        self.dir_name = dir_name
        self.file_name = os.path.join(self.dir_name, data_file)

    def load_format_data(
        self,
        load_source="auto",
        query=None,
        is_update_local_file=False,
        col_time=None,
        query_file=None,
    ):
        self.load_source = load_source
        self._set_load_source()
        self._get_query(query, query_file)
        self._load_data()
        if is_update_local_file:
            if self.df.empty:
                print(
                    ("Retrieved dataframe is empty. " "Not going to update local file")
                )
                sys.exit(1)
            if not self.file_name:
                print(
                    (
                        "Run self._set_file_name(dir_name, data_file) "
                        ",to set output path. Not going to update local file"
                    )
                )
            else:
                self._save_data()
        self._set_time_col(col_time)


def sample_read(out, n, s, seed=42):
    random.seed(seed)
    skip = sorted(random.sample(np.xrange(1, n + 1), n - s))
    return pd.read_csv(out, skiprows=skip)
