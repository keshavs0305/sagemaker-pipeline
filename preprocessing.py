import argparse
import os
import requests
import tempfile

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder


if __name__ == "__main__":
    base_dir = "/opt/ml/processing"

    df = pd.read_csv(
        f"{base_dir}/input/features.csv",
#        header=None,
#        names=feature_columns_names + [label_column],
#        dtype=merge_two_dicts(feature_columns_dtype, label_column_dtype),
    )
    df.drop(['key'],axis=1,inplace=True)
    features = list(df.drop(['feature76'],axis=1).columns)
    #numeric_features.remove("sex")
    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )

    #categorical_features = ["sex"]
    #categorical_transformer = Pipeline(
    #    steps=[
    #        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
    #        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    #    ]
    #)

    preprocess = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, features)
    #        ("cat", categorical_transformer, categorical_features),
        ]
    )

    y = df.pop("feature76")
    X_pre = preprocess.fit_transform(df)
    y_pre = y.to_numpy().reshape(len(y), 1)

    X = np.concatenate((y_pre, X_pre), axis=1)

    np.random.shuffle(X)
    train, validation, test = np.split(X, [int(0.7 * len(X)), int(0.85 * len(X))])

    pd.DataFrame(train).to_csv(f"{base_dir}/train/train.csv", header=False, index=False)
    pd.DataFrame(validation).to_csv(
        f"{base_dir}/validation/validation.csv", header=False, index=False
    )
    pd.DataFrame(test).to_csv(f"{base_dir}/test/test.csv", header=False, index=False)
