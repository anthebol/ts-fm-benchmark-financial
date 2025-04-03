import numpy as np
import pandas as pd


class Normalizer(object):
    """
    Normalizes dataframe across ALL contained rows (time steps). Different from per-sample normalization.
    """

    def __init__(
        self,
        norm_type="standardization",
        mean=None,
        std=None,
        min_val=None,
        max_val=None,
    ):
        self.norm_type = norm_type
        self.mean = mean
        self.std = std
        self.min_val = min_val
        self.max_val = max_val

    def normalize(self, df):
        if self.norm_type == "standardization":
            if self.mean is None:
                self.mean = df.mean()
                self.std = df.std()
            return (df - self.mean) / (self.std + np.finfo(float).eps)

        elif self.norm_type == "minmax":
            if self.max_val is None:
                self.max_val = df.max()
                self.min_val = df.min()
            return (df - self.min_val) / (
                self.max_val - self.min_val + np.finfo(float).eps
            )

        else:
            raise NameError(f'Normalize method "{self.norm_type}" not implemented')

    def denormalize(self, df):
        if isinstance(df, np.ndarray):
            df = pd.DataFrame(df)

        if self.norm_type == "standardization":
            return (df * self.std) + self.mean

        elif self.norm_type == "minmax":
            return df * (self.max_val - self.min_val) + self.min_val

        else:
            raise NameError(f'Denormalize method "{self.norm_type}" not implemented')
