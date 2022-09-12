"""
# DSSP: Data Science Job Salary Linear Predictor

## Data
Source: https://www.kaggle.com/datasets/ruchi798/data-science-job-salaries

### Citation
> None
"""

# Data Loading and Project Preparation

# Commented out IPython magic to ensure Python compatibility.
# %pip install forex-python

try:
    # utils
    import pandas as pd
    import numpy as np
    import shutil
    from zipfile import ZipFile
    import os
    from IPython.display import display
    import json
    import pickle

    from forex_python.converter import CurrencyRates

    from sklearn.model_selection import train_test_split
    from sklearn.decomposition import PCA

    # plotting tools
    import matplotlib
    from matplotlib import pyplot as plt
    import seaborn as sns
    from keras.utils.vis_utils import plot_model

    # modeling
    import tensorflow as tf
    import keras

    from keras import Sequential
    from keras.layers import add
    from keras.layers import Normalization, Dropout, Dense

    from sklearn.metrics import mean_squared_error

    # creating a baseline
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.model_selection import GridSearchCV

    print(f"Using tensorflow v{tf.__version__}")
    print(f"Using keras v{keras.__version__}")
    print(f"Using matplotlib v{matplotlib.__version__}")
except Exception:
    import sys

    exc_type, exc_obj, exc_tb = sys.exc_info()
    raise Exception(
        f"error on importing dependencies! Error on line: {exc_tb.tb_lineno}"
    )
finally:
    print("\nsuccess on importing dependencies!")


class utils:
    def kaggle(token_path: str, datasets=None, copy=True):
        root_dir = "/root/.kaggle"
        file_name = "kaggle.json"
        file_loc = os.path.join(root_dir, file_name)

        if not os.path.isdir(root_dir):
            os.mkdir(root_dir)

        if copy:
            shutil.copy(token_path, root_dir)
            status = "copied"
        else:
            shutil.move(token_path, root_dir)
            status = "moved"

        os.chmod(file_loc, 600)
        print(f"'kaggle.json' file have been {status} to {file_loc}!")

        if datasets is not None:
            from kaggle.api.kaggle_api_extended import KaggleApi

            api = KaggleApi()
            api.authenticate()

            if type(datasets) == str:
                datasets = [datasets]

            for dataset in datasets:
                api.dataset_download_cli(dataset)

            print(f"Dataset have been successfully downloaded!")

    def unzip(path: str, output_dir: str):
        with ZipFile(path, "r") as zf:
            zf.extractall(output_dir)

        print(f"'{path}' have been extracted!")


utils.kaggle(
    token_path="drive/MyDrive/api_identifiers/kaggle.json",
    datasets="ruchi798/data-science-job-salaries",
)

utils.unzip(path="data-science-job-salaries.zip", output_dir=os.getcwd())

df = pd.read_csv("ds_salaries.csv")

# Data Preparation

df.drop(labels=["Unnamed: 0"], axis=1, inplace=True)

# Data Transformation


def get_mapping(df):
    columns = df.columns.tolist()
    num_mappings = dict()

    for c in columns:
        if df[c].dtype in [object, str]:
            data = df[c].unique()
            obj_mappings = dict()

            for i in range(len(data)):
                obj_mappings[data[i]] = i

            num_mappings[c] = obj_mappings

    return num_mappings


def transform(df, map):
    for column, obj_map in map.items():
        for obj, code in obj_map.items():
            df.replace(to_replace=obj, value=code, inplace=True)


def get_usd_salary(df):
    d1 = df["salary"].values.tolist()
    d2 = df["salary_currency"].values.tolist()
    b1 = df["salary_in_usd"].values.tolist()
    usd_d1 = list()

    c = CurrencyRates()

    for index in range(len(d1)):
        try:
            rate = c.get_rate(d2[index], "USD")
            usd_d1.append(int(round(rate * d1[index])))
        except Exception:
            usd_d1.append(b1[index])

    return pd.DataFrame(usd_d1, columns=["salary_in_usd"])


usd = get_usd_salary(df)

columns = df.columns.tolist()

s1 = df[columns[:7]]
s2 = df[columns[7:]]

s1.drop(labels=["salary_in_usd"], axis=1, inplace=True)

s1 = pd.concat([s1, usd], axis=1)
df = pd.concat([s1, s2], axis=1)

print(df.info())

mapp = get_mapping(df)

transform(df, mapp)

with open("object_mappings.json", "w") as json_file:
    json.dump(mapp, json_file)

# Feature Selection

print("salary currencies:\n{}".format(mapp["salary_currency"]))
print("employee residences:\n{}".format(mapp["employee_residence"]))

df.drop(labels=["salary", "salary_currency"], axis=1, inplace=True)

df.drop(labels=["work_year"], axis=1, inplace=True)

# Dimensionality Reduction

columns = df.columns.tolist()

y_column = "salary_in_usd"
columns.remove(y_column)
x_columns = columns

x = df[x_columns]
y = df[[y_column]]

fts = x.columns.tolist()

pca = PCA(n_components=1, random_state=123)
pca.fit(x[fts])

x["features"] = pca.transform(x.loc[:, tuple(fts)]).flatten()
x.drop(labels=fts, axis=1, inplace=True)

# Train Test Split


def gss(dataset_length, test_size, val_size):
    test = round(dataset_length * test_size)
    train = dataset_length - test
    val = round(train * val_size)

    train = train - val

    return (train, test, val)


sz = [0.05, 0.1, 0.2, 0.3, 0.4]

for size in sz:
    for val_size in sz:
        train, test, val = gss(len(df), size, val_size)

        print(
            f"dataset training size for test size of {size} and validation size of {val_size}: \n\ttrain data: {train}\n\ttest data: {test}\n\tvalidation data: {val}"
        )

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)

display(x_train, y_train, x_test, y_test)

# Modelling

# Creating a Basline


def create_baseline(x, y):
    baseline_model = KNeighborsRegressor()
    aval_params = [
        {
            "n_neighbors": (range(10, 50)),
            "algorithm": ("ball_tree", "kd_tree", "brute"),
            "metric": ("minkowski", "euclidean"),
        }
    ]

    gridsearch = GridSearchCV(
        baseline_model, aval_params, cv=15, scoring="neg_mean_squared_error"
    )
    gridsearch.fit(x, y)

    best_params = gridsearch.best_params_

    baseline_model = KNeighborsRegressor(
        algorithm=best_params["algorithm"],
        metric=best_params["metric"],
        n_neighbors=best_params["n_neighbors"],
    )

    baseline_model.fit(x, y)
    print(f"baseline score: {gridsearch.best_score_}")

    return baseline_model


baseline = create_baseline(x_train, y_train)

# Model Development

model = Sequential()

model.add(Dense(64, activation="relu", kernel_initializer="normal"))
model.add(Dense(32, activation="relu", kernel_initializer="normal"))
model.add(Dropout(0.6))
model.add(Dense(1, kernel_initializer="normal"))

model.compile(
    loss="mean_squared_error",
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    metrics=["mse"],
)

model.build(input_shape=(None, 1))

history = model.fit(
    x_train.values, y_train.values, validation_split=0.2, epochs=232, verbose=0
)

# Evaluation


def get_result(model, x, y):
    y_pred = model.predict(x).flatten()
    y = y.reset_index()
    x = x.reset_index()

    y.drop(labels=["index"], axis=1, inplace=True)
    x.drop(labels=["index"], axis=1, inplace=True)

    y_hats = list()

    for y_hat in y_pred:
        y_hats.append(round(y_hat))

    result = pd.concat([x, y, pd.DataFrame(y_hats, columns=["y_hat"])], axis=1)

    return result


def get_report(model_dict, x_train, y_train, x_test, y_test):
    mse = pd.DataFrame(columns=["train", "test"], index=model_dict.keys())

    for name, model in model_dict.items():
        mse.loc[name, "train"] = (
            mean_squared_error(y_true=y_train, y_pred=model.predict(x_train)) / 1e3
        )
        mse.loc[name, "test"] = (
            mean_squared_error(y_true=y_test, y_pred=model.predict(x_test)) / 1e3
        )

    return mse


model_dict = {"baseline_knn": baseline, "neural_networks": model}

report1 = get_report(model_dict, x_train, y_train, x_test, y_test)
print(report1)

# Saving Both Models

model.save("/content/predictor.h5")

with open("knn_predictor.pkl", "wb") as f:
    pickle.dump(baseline, f)
