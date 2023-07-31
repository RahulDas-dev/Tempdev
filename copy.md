import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.ensemble import RandomForestClassifier

numerical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

def from_bool_to_number(x: np.ndarray) -> np.ndarray:
    return np.where(x, 1.0, 0.0)


def from_number_to_bool(x: np.ndarray) -> np.ndarray:
    return np.where(x > 0.5, True, False)


BooleanTransformer = FunctionTransformer(from_bool_to_number, from_number_to_bool)

# Combine the preprocessing pipelines using ColumnTransformer
preprocessor = ColumnTransformer([
    (
        'numerical', 
         Pipeline([
                ('imputer', SimpleImputer(strategy='mean')),
                #('scaler', StandardScaler())
         ]), 
         column_selector(dtype_include=[np.number]) 
    ),
    (
        'categorical', 
          Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
          ]), 
          column_selector(dtype_include=[object, "category"]) 
    ),
    (
        'boolean', 
          Pipeline([
                ('imputer', ReplaceInf()),
                ('onehot', FunctionTransformer(from_bool_to_number, from_number_to_bool, check_inverse=False))
          ]), 
          column_selector(dtype_include=bool)
    )
    
])

# Add the classifier to the pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor ),
    ('classifier', KNeighborsClassifier() )
])

train = data





from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.utils import check_array

class ReplaceInf(TransformerMixin, BaseEstimator):

    def fit(self, X, y=None):
        # validate and convert if possible:
        X = check_array(X, force_all_finite=False)
        _, counts = np.unique(X, return_counts=True)
        ind = np.argmax(counts)
        self.fill_val_ = X[ind]
        return self

    def transform(self, X):
        X = check_array(X, force_all_finite=False)
        return np.where(X==np.nan, self.fill_val_, X)

target = dataset.pop('class')
pipeline.fit(dataset, target)

z = pipeline.predict(dataset, target)




from sklearn.datasets import  fetch_california_housing,load_diabetes,fetch_openml
#1461,31,29
x = fetch_openml(data_id=1461, as_frame=True, parser='pandas')



