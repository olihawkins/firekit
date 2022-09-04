"""
Functions for preprocessing data.
"""

# Imports ---------------------------------------------------------------------

import numpy as np

# Split dataframe -------------------------------------------------------------

def split_dataframe(
    df,
    train_size,
    val_size):

    # Get indices for splits
    train_end = np.floor(train_size * df.shape[0]).astype(int)
    val_end = np.floor((train_size + val_size) * df.shape[0]).astype(int)
    
    # Split into train, validation and test dataframes
    train_df = df.iloc[:train_end, :]
    val_df = df.iloc[train_end:val_end, :]
    test_df = df.iloc[val_end:, :]

    return train_df, val_df, test_df