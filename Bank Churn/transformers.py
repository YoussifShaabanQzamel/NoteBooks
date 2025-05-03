
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer, KNNImputer
import category_encoders as ce

from .transformers import (
    StandardScaleTransform,
    CustomOneHotEncoder,
    OrdinalEncodeColumns,
    DataFrameImputer,
    TargetEncoderTransformer
)

__all__ = [
    'StandardScaleTransform',
    'CustomOneHotEncoder',
    'OrdinalEncodeColumns',
    'DataFrameImputer',
    'TargetEncoderTransformer'
]
class StandardScaleTransform(BaseEstimator, TransformerMixin):
    """
    A transformer class to apply standard scaling to specified columns in a Pandas DataFrame.

    Parameters
    ----------
    cols : list of str
        The names of the columns to apply standard scaling to.
    """
    def __init__(self, cols):
        self.cols = cols
        self.scaler_ = None

    def fit(self, X, y=None):
        self.scaler_ = StandardScaler().fit(X.loc[:, self.cols])
        return self

    def transform(self, X):
        X_copy = X.copy()
        X_copy.loc[:, self.cols] = self.scaler_.transform(X_copy.loc[:, self.cols])
        return X_copy

    def fit_transform(self, X, y=None):
        self.scaler_ = StandardScaler().fit(X.loc[:, self.cols])
        return self.transform(X)

class CustomOneHotEncoder(BaseEstimator, TransformerMixin):

    """
    A transformer class to apply one-hot encoding to specified columns in a Pandas DataFrame.

    Parameters
    ----------
    columns : list
        A list of column names to encode.

    Returns
    -------
    pandas.DataFrame
        A new DataFrame with the specified columns one-hot encoded.
    """
    def __init__(self, columns=None):
        self.columns = columns
        self.feature_names_ = None

    def fit(self, X, y=None):
        self.feature_names_ = pd.get_dummies(X[self.columns], prefix=self.columns).columns.tolist()
        return self

    def transform(self, X):
        X_transformed = pd.get_dummies(X[self.columns], prefix=self.columns)
        
        # Drop original columns if necessary
        if self.columns is not None:
            X = X.drop(self.columns, axis=1)
        
        # Concatenate transformed columns with remaining DataFrame
        X = pd.concat([X, X_transformed], axis=1)
        return X
    
    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)

class OrdinalEncodeColumns(BaseEstimator, TransformerMixin):
    """
    Transformer class to perform ordinal encoding on specified columns of a Pandas DataFrame.

    Parameters
    ----------
    columns : list of str
        The names of the ordinal columns to encode.

    Returns
    -------
    pandas.DataFrame
        A new DataFrame with the ordinal columns encoded.
    """
    def __init__(self, columns):
        self.columns = columns
        self.encoder = None
    
    def fit(self, X, y=None):
        ordinal_data = X[self.columns].values
        self.encoder = OrdinalEncoder()
        self.encoder.fit(ordinal_data)
        return self
    
    def transform(self, X):
        X_new = X.copy()
        ordinal_data = X_new[self.columns].values
        encoded_data = self.encoder.transform(ordinal_data)
        X_new[self.columns] = encoded_data
        return X_new
    
    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)

class DataFrameImputer(TransformerMixin, BaseEstimator):
    """
    A class to impute missing values in a Pandas DataFrame using a combination of median, knn, and most frequent
    imputers on specified columns.

    Parameters:
    -----------
    median_cols : list of str, optional (default=None)
        Columns to impute missing values using the median imputer.
    knn_cols : list of str, optional (default=None)
        Columns to impute missing values using the KNN imputer.
    freq_cols : list of str, optional (default=None)
        Columns to impute missing values using the most frequent imputer.
    const_cols : dict of {column_name: constant_value} pairs, optional (default=None)
        Columns to impute missing values using a constant value.

    Returns:
    --------
    X_imputed : pandas.DataFrame
        A DataFrame with imputed missing values.
    """
    def __init__(self, median_cols=None, knn_cols=None, freq_cols=None, const_cols=None, fill_const=0):
        self.median_cols = median_cols
        self.knn_cols = knn_cols
        self.freq_cols = freq_cols
        self.const_cols = const_cols
        self.fill_const = fill_const
    
    def fit(self, X, y=None):
        self.median_imputer = SimpleImputer(strategy='median')
        self.knn_imputer = KNNImputer()
        self.freq_imputer = SimpleImputer(strategy='most_frequent')
        self.const_imputer = SimpleImputer(strategy='constant', fill_value=self.fill_const)

        if self.median_cols is not None:
            self.median_imputer.fit(X[self.median_cols])
        if self.knn_cols is not None:
            self.knn_imputer.fit(X[self.knn_cols])
        if self.freq_cols is not None:
            self.freq_imputer.fit(X[self.freq_cols])
        if self.const_cols is not None:
            self.const_imputer.fit(X[self.const_cols])

        return self
    
    def transform(self, X):
        X_imputed = X.copy()
        if self.median_cols is not None:
            X_median = pd.DataFrame(self.median_imputer.transform(X[self.median_cols]), 
                                    columns=self.median_cols, index=X.index)
            X_imputed = pd.concat([X_imputed.drop(self.median_cols, axis=1), X_median], axis=1)
        if self.knn_cols is not None:
            X_knn = pd.DataFrame(self.knn_imputer.transform(X[self.knn_cols]), 
                                 columns=self.knn_cols, index=X.index)
            X_imputed = pd.concat([X_imputed.drop(self.knn_cols, axis=1), X_knn], axis=1)
        if self.freq_cols is not None:
            X_freq = pd.DataFrame(self.freq_imputer.transform(X[self.freq_cols]), 
                                  columns=self.freq_cols, index=X.index)
            X_imputed = pd.concat([X_imputed.drop(self.freq_cols, axis=1), X_freq], axis=1)
        if self.const_cols is not None:
            X_const = pd.DataFrame(self.const_imputer.transform(X[self.const_cols]), 
                                  columns=self.const_cols, index=X.index)
            X_imputed = pd.concat([X_imputed.drop(self.const_cols, axis=1), X_const], axis=1)
        return X_imputed
    
    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)

class TargetEncoderTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, cols=None):
        self.cols = cols  # List of categorical columns to encode
        self.encoder = None  # Placeholder for the category encoder

    def fit(self, X, y):
        """Fit the target encoder using the training data."""
        self.encoder = ce.TargetEncoder(cols=self.cols)
        self.encoder.fit(X[self.cols], y)
        return self  # Return the transformer object

    def transform(self, X):
        """Apply the target encoding to the data (test or new data)."""
        X_transformed = X.copy()
        X_transformed[self.cols] = self.encoder.transform(X[self.cols])
        return X_transformed
