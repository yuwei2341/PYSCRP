#!/usr/bin/env python3

# Lib for preprocessing

import pandas as pd

# FUNCTIONS
def set_timeind_slice(df, time_col, tz_to, tz_from='UTC', start= '1970-01-01', end='2050-12-31'):
    '''Set timestamp index, convert time zone and slice

    df: the data frame
    time_col: timestamp column to assign to index
    tz_to: timezone to convert to
    tz_from: timezone of original time; 'UTC' by default
    start, end: start and end date to slice, default set to be large so as to get the whole df

    '''

    df.index = pd.to_datetime(df[time_col])
    df = df.tz_localize(tz_from).tz_convert(tz_to).sort_index()
    df = df.loc[pd.Timestamp(start, tz=tz_to):pd.Timestamp(end, tz=tz_to)].sort_index()

    return df

def get_count_perc(x, rm_zero, dropna):
    '''Get value counts and percentage of each value in total '''
    if rm_zero:
        x = x[x != 0]
    df_count = x.value_counts(dropna=dropna)
    df_count_perc = pd.concat([df_count,
                               (x.value_counts(normalize=True, dropna=dropna) * 100).rename(df_count.name + '%')], axis=1)
    df_count_perc.loc['Total'] = df_count_perc.sum()
    return df_count_perc

def value_counts_plus(df, rm_zero=False, dropna=False):
    return pd.concat([get_count_perc(df[ii], rm_zero, dropna) for ii in df], axis=1).fillna('').astype(str)
