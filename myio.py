#!/usr/bin/env python
import os
import sys
import random
import socket

import pandas as pd

from queryrunner_client import Client
# parameters for query_runner
QR_PARAMS = {'user_email': 'yuwei@uber.com'}
if socket.gethostname() == 'yuwei-C02SH0TQG8WL':
    QR_PARAMS['hdfs_proxy_port'] = 9999
QR = Client(**QR_PARAMS)

class DataIO(object):
    '''Read data as pandas dataframe from local file or query '''
    def __init__(self):
        self.dir_name = ''
        self.data_file = ''
        self.file_name = ''
        self.query = None
        self.df = None
        self.load_source = ''

    def _set_load_source(self):
        if self.load_source not in ('auto', 'file', 'query'):
            print 'Wrong load source'
            sys.exit(1)
        if self.load_source == 'auto':
            self.load_source = 'file' if os.path.exists(self.file_name) else 'query'

    def _load_data(self, is_update_local_file):
        if self.load_source == 'file':
            self.df = pd.read_csv(self.file_name)
            print 'Local data loaded'
        else:
            print 'Querying data... '
            if self.query is None:
                print 'No query provided'
                sys.exit(1)
            data = QR.execute(self.query['db'], self.query['query'], timeout=10**6).fetchall()
            self.df = pd.DataFrame(data)
            if is_update_local_file:
                self.df.to_csv(self.file_name, index=False)
                print 'Local data saved/updated from query'

    def _set_time_col(self, col_time):
        if col_time is not None and self.df[col_time].dtype.kind != 'M':
            self.df[col_time] = pd.to_datetime(self.df[col_time])

    def set_dir(self, dir_name):
        self.dir_name = dir_name

    def load_format_data(self, load_source='auto', data_file='', query=None, is_update_local_file=True, col_time=None):
        self.load_source = load_source
        self.data_file = data_file
        self.file_name = os.path.join(self.dir_name, self.data_file)
        self.query = query
        self._set_load_source()
        self._load_data(is_update_local_file)
        self._set_time_col(col_time)

def sample_read(out, n, s, seed=42):
	random.seed(seed)
	skip = sorted(random.sample(xrange(1, n + 1), n - s))
	return pd.read_csv(out, skiprows=skip)
