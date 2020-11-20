
# %% Preliminaries

# Import
import pandas as pd
import numpy as np
from tqdm import tqdm
import math

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator, TransformerMixin

from imblearn.over_sampling import SMOTE

# Paths
MAIN_PATH = r"/Users/paulmora/Documents/projects/insurance_claim"
RAW_PATH = rf"{MAIN_PATH}/00 Raw"
CODE_PATH = rf"{MAIN_PATH}/01 Code"
DATA_PATH = rf"{MAIN_PATH}/02 Data"
OUTPUT_PATH = rf"{MAIN_PATH}/03 Output"

# Data
train_data = pd.read_csv(rf"{RAW_PATH}/train.csv")
test_data = pd.read_csv(rf"{RAW_PATH}/test.csv")
total_data = pd.concat([train_data, test_data], axis=0)
total_data.reset_index(drop=True, inplace=True)

### SMALL DATASET
total_data = total_data.loc[:1_000, :]

# %% Settings

# Matplotlib sizes
SMALL_SIZE = 15
MEDIUM_SIZE = 20
BIGGER_SIZE = 30

plt.rc("font", size=SMALL_SIZE)
plt.rc("axes", titlesize=SMALL_SIZE)
plt.rc("axes", labelsize=MEDIUM_SIZE)
plt.rc("xtick", labelsize=SMALL_SIZE)
plt.rc("ytick", labelsize=SMALL_SIZE)
plt.rc("legend", fontsize=SMALL_SIZE)
plt.rc("figure", titlesize=BIGGER_SIZE)

# %% Missing observations

"""
For that we start by replacing the cells with -1 with nan. Afterwards we plot the number of missing observations
to get an overall feeling of the situation
"""

total_data.replace(-1, np.nan, inplace=True)
total_data_wo_target = total_data.drop(columns="target")
missing_pct_series = total_data_wo_target.apply(lambda x: (sum(x.isna()) / len(x)) * 100)
sorted_missing_pct = missing_pct_series.sort_values(ascending=False)
fig, axs = plt.subplots(figsize=(20, 10))
sorted_missing_pct.plot(kind="bar", ax=axs)
path = rf"{OUTPUT_PATH}/missing_obs.png"
fig.savefig(path, bbox_inches="tight")
plt.close()

"""
We find two variables which have either 50% or even 70% missing observations. The latter variable will definitely
be dropped given the high amount of missing data. For the second we quickly check the correlation with the
target variable in order to see whether trying to rescue it is worth it.

Furthermore we note down which columns do not have any missing values. These are important, given that we can
use them in predictive models for the target variable.
"""

THRESHOLD_MISSING = 40
highly_missing_columns = list(sorted_missing_pct[sorted_missing_pct > THRESHOLD_MISSING].index)
highly_missing_corr_df = train_data.loc[:, highly_missing_columns + ["target"]].corr()
total_data.drop(columns=highly_missing_columns, inplace=True)

# %% Allocate variable to correct category


def categorical_check(column, data):
    series = data.loc[:, column].dropna()
    bool_series_cat = series.value_counts().min() > 1
    bool_binary = list(set(series) - set([0, 1])) == []
    return bool_series_cat and not bool_binary


def binary_check(column, data):
    series = data.loc[:, column].dropna()
    bool_binary = list(set(series) - set([0, 1])) == []
    return bool_binary


def float_check(column, data):
    series = data.loc[:, column].dropna()
    bool_float = series.value_counts().min() == 1
    return bool_float


y = total_data.loc[:, "target"]
feature_data = total_data.drop(columns=["target", "id"])
cat_variables = [x for x in feature_data.columns if categorical_check(x, feature_data)]
bin_variables = [x for x in feature_data.columns if binary_check(x, feature_data)]
float_variables = [x for x in feature_data.columns if float_check(x, feature_data)]

assert bool(set(cat_variables) & set(bin_variables) & set(float_variables)) is False, "Multiple Columns in diff Cat"

# %% Imputation Classes

# Categorical
class CategoricalImputation:

    def __int__(self, columns):
        self.columns = columns

    def columns_divider(self, X):
        """This method divides the columns with nans and those which do not have any"""
        bool_missing = X.isna().any()
        columns_w_nans = X.columns[bool_missing]
        columns_wo_nans = X.columns[~bool_missing]
        return columns_w_nans, columns_wo_nans

    def

    def fit(self, X, y=None):
        features, target = self.columns_divider(X)
        neigh = NearestNeighbors(n_neighbors=1)
        neigh.fit(features, target)



a = CategoricalImputation()

a.fit(feature_data)

feature_data.columns[feature_data.isna().any()]

# Floats


# Binaries


features = list(set(total_data) - set(["id", "target"]))
df_features = total_data.loc[:, features]

minmax_scaler = MinMaxScaler()
imputer = KNNImputer(n_neighbors=1, weights="uniform", metric="nan_euclidean")
pipe = Pipeline([("scaler", minmax_scaler), ("imputation", imputer)])
pipe.fit(df_features)

scaled_total_data = total_data.copy()
scaled_total_data.loc[:, features] = pipe.transform(df_features)

assert scaled_total_data.isna().any().any() == False, "Still Missing Data"

# %% Target Variable - Imbalanced Data

"""
Before moving on to the feature creation we take a gander on the balancing of the class
"""


def count_plot_creation(data, name):
    """
    This function plots the count of the target variable
    :param data: dataframe: which contains a column called target
    :param name: str: String to specify how the plot should be saved
    """
    fig, axs = plt.subplots(figsize=(10, 10))
    sns.countplot(x="target", data=data, ax=axs)
    axs.tick_params(axis="both", which="major")
    axs.set_ylabel("Count")
    path = rf"{OUTPUT_PATH}/count_plot_{name}.png"
    fig.savefig(path, bbox_inches='tight')
    plt.close()


def pca_scatterplot_creation(data, features, name):

    pca = PCA(n_components=2)
    pca_factors = pd.DataFrame(pca.fit_transform(data.loc[:, features]),
                               columns=["PCA_0", "PCA_1"])
    pca_factors.loc[:, "target"] = data.loc[:, "target"]
    fig, axs = plt.subplots(figsize=(10, 10))
    sns.scatterplot(data=pca_factors, x="PCA_0", y="PCA_1", hue="target")
    path = rf"{OUTPUT_PATH}/scatter_plot_{name}.png"
    fig.savefig(path, bbox_inches='tight')
    plt.close()


count_plot_creation(scaled_total_data, "no_sampling")
pca_scatterplot_creation(scaled_total_data, features, "no_sampling")


"""
As in every classification project within insurance we face quite a strong imbalance. To mitigate that problem
we fire up a SMOTE over-sampler. We try 
"""

oversample = SMOTE(sampling_strategy=0.5)
x_data = scaled_total_data.loc[:, features]
y_data = scaled_total_data.loc[:, "target"]
sampled_x_data, sampled_y_data = oversample.fit_resample(x_data, y_data)
upsampled_total_data = pd.concat([sampled_x_data, sampled_y_data], axis=1)

pca_scatterplot_creation(upsampled_total_data, features, "upsampling")

# %% Feature Engineering

# Re-assigning the features into different categories

"""
Within the data we have several variables which are indicated to be floats, even though they should be
classified as categorical given that all observations fall into a category which is only labelled as a
numeric value
"""

for feature in features:
    bool_float_cat = upsampled_total_data.loc[:, feature].value_counts().min() > 1
    bool_float_indication = feature[-1].isnumeric()
    if bool_float_cat and bool_float_indication:
        upsampled_total_data.rename(columns={feature: f"{feature}_float_cat"}, inplace=True)

# Float variables

"""
Before diving into the variables we first have to change the type of some numeric columns. That is because
we have some variables which are denoted as float, but are actually categorical. That can be seen when looking
at the following plot of float columns
"""

float_columns = [x for x in upsampled_total_data.columns if x[-1].isnumeric()]
float_plot_cols = 2
float_plot_rows = math.ceil(len(float_columns) / float_plot_cols)
fig, axs = plt.subplots(nrows=float_plot_rows, ncols=float_plot_cols,
                        figsize=(float_plot_cols * 10, float_plot_rows * 10))
axs = axs.ravel()
for i, float_column in tqdm(enumerate(float_columns)):
    sns.histplot(upsampled_total_data.loc[:, float_column], ax=axs[i])
path = rf"{OUTPUT_PATH}/continuous_variables_dist.png"
fig.savefig(path, bbox_inches='tight')
plt.close()

# Categorical variables

"""
We therefore define every column of these float variables as a categorical if every category has more than one
observation. This would not be the case if the variable would be a continuous variable
"""

for float_column in float_columns:
    if total_data.loc[:, float_column].value_counts().min() > 1:
        total_data.rename(columns={float_column: f"{float_column}_float_cat"}, inplace=True)

"""
Categorical Variables

These variables are encoded with the mean of the target data. It is necessary before doing that to see that we
have all categories we find in train also in test
"""


def mean_encoding(encoded_column, df_data):

    mean_encoded_variables = df_data.groupby([encoded_column])["target"].mean()
    encoded_variable = df_data.loc[:, encoded_column].map(mean_encoded_variables)
    return encoded_variable


for cat_column in tqdm(cat_variables):
    total_data.loc[:, cat_column] = mean_encoding(cat_column, total_data)


total_data.loc[:, "ps_car_11"]

"""
From the plots we do not see any significant outliers we would have to adjust.
"""

total_data.loc[:, "ps_car_11"].value_counts()

a = total_data.loc[:, "ps_car_13"].value_counts()


# %% Model building





# %%



