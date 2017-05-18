#!/usr/bin/env python

import random

def sample_read(out, n, s, seed=42):
	random.seed(seed)
	skip = sorted(random.sample(xrange(1, n + 1), n - s))
	return pd.read_csv(out, skiprows=skip)
