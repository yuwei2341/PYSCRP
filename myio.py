#!/usr/bin/env python

import random
import socket

import pandas as pd

# parameters for query_runner
qr_params = {'user_email': 'yuwei@uber.com'}
if socket.gethostname() == 'yuwei-C02SH0TQG8WL':
    qr_params['hdfs_proxy_port'] = 9999

def sample_read(out, n, s, seed=42):
	random.seed(seed)
	skip = sorted(random.sample(xrange(1, n + 1), n - s))
	return pd.read_csv(out, skiprows=skip)

