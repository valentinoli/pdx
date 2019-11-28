# -*- coding: utf-8 -*-
"""Various helper functions"""


def df_to_tril(df):
    """Returns a lower triangular dataframe where entries
    above and including the main diagonal are set to zero.
    """
    for index, row in df.iterrows():
        row[index <= row.index] = 0
    return df

