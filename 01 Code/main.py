
# %% Preliminaries

# Import
import pandas as pd
import numpy as np
from tqdm import tqdm
import math

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler

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

"""
Given the very small correlation we see between the highly missing columns and the target we decide to simply drop
them since it is not worth trying to rescue them.
"""

new_sorted_missing_pct = sorted_missing_pct.drop(highly_missing_columns)
no_nan_columns = list(sorted_missing_pct[sorted_missing_pct == 0].index)
still_missing_columns = list(set(sorted_missing_pct.index) - set(no_nan_columns))

"""
Given that we found several variables which show that they are floats even though they are binary variables, this
is checked and changed where needed.
"""

features = list(set(total_data.columns) - set(["id", "target"]))

feature = "ps_ind_04_cat"

for feature in tqdm(features):
    bool_nan_series = total_data.loc[:, feature].isna()
    non_nan_series = total_data.loc[~bool_nan_series, feature]
    bool_already_binary_column = feature.endswith("_bin")
    bool_binary_content = set(non_nan_series) - set([0, 1]) == set()
    if bool_binary_content and not bool_already_binary_column:
        total_data.loc[~bool_nan_series, feature] = total_data.loc[~bool_nan_series, feature].astype(int)
        total_data.rename(columns={feature: f"{feature}_bin"}, inplace=True)
# NOT WORKING

"""
Furthermore we find a couple of variables which are indicated as floats but they are more like categorical variables,
given
"""

    bool_float_cat = total_data.loc[:, feature].value_counts().min() > 1
    bool_float_indication = feature[-1].isnumeric()
    if bool_float_cat and bool_float_indication:
        total_data.rename(columns={feature: f"{feature}_float_cat"}, inplace=True)

total_data.loc[:, "ps_car_07_cat"].value_counts()

feature = features[0]

cat_vars
float_vars =

float_columns = [x for (x, y) in zip(total_data.columns, total_data.dtypes) if y == float and x[-1].isnumeric()]
cat_variables = [x for (x, y) in zip(total_data.columns, total_data.dtypes) if x.endswith("_cat") or y == int]

total_data.loc[:, "ps_ind_07_bin"]

"""
For the moment we simply will fill in the mean before doing something more sophisticated

Ideas:
    - KNN for categorical
    - KNN Regression for continuous variables

"""

total_data.drop(columns=highly_missing_columns, inplace=True)
filled_total_data = total_data.fillna(total_data.mean())

assert filled_total_data.isna().any().any() == False, "Still Missing Data"

# %% Target Variable - Imbalanced Data

"""
As in every classification project within insurance we face quite a strong imbalance. In the first go we will not
address this problem but will come back to this and try:
    - SMOTE Up-sampling
    - SMOTE Up-sampling plus random down-sampling
"""

fig, axs = plt.subplots(figsize=(10, 10))
sns.countplot(x="target", data=total_data, ax=axs)
axs.tick_params(axis="both", which="major")
axs.set_ylabel("Count")
path = rf"{OUTPUT_PATH}/count_plot.png"
fig.savefig(path, bbox_inches='tight')
plt.close()

# %% Feature Engineering

"""
Before diving into the variables we first have to change the type of some numeric columns. That is because
we have some variables which are denoted as float, but are actually categorical. That can be seen when looking
at the following plot of float columns
"""

float_plot_cols = 2
float_plot_rows = math.ceil(len(float_columns) / float_plot_cols)
fig, axs = plt.subplots(nrows=float_plot_rows, ncols=float_plot_cols,
                        figsize=(float_plot_cols * 10, float_plot_rows * 10))
axs = axs.ravel()
for i, float_column in tqdm(enumerate(float_columns)):
    sns.histplot(total_data.loc[:, float_column], ax=axs[i])
path = rf"{OUTPUT_PATH}/continuous_variables_dist.png"
fig.savefig(path, bbox_inches='tight')
plt.close()

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


# %%


def mean_imputing(data, target):

    mean_of_target_variable = np.nanmean(data.loc[:, target])
    filled_column = data.loc[:, target].fillna(mean_of_target_variable)
    return filled_column


def median_imputing(data, target):

    median_of_target_variable = np.nanmedian(data.loc[:, target])
    filled_column = data.loc[:, target].fillna(median_of_target_variable)
    return filled_column


def nearest_neighbor(data, target, features)

    bool_nan_target_rows = data.loc[:, target].isna()

    x_raw = data.loc[~bool_nan_target_rows, features]
    x_norm = MinMaxScaler().fit_transform(x_raw)
    y = data.loc[~bool_nan_target_rows, target]

    neigh = KNeighborsRegressor(n_neighbors=1)
    neigh.fit(x_norm, y)

    neigh.predict(x_norm[bool_nan_target_rows])
    neigh.predict(data.loc[~bool_nan_target_rows, features])



def imputing_assessment(imputed_series):


    time_series = imputed_series.dropna().reset_index(drop=True)

    n = 50
    random.seed(42)
    rand_num = random.sample(range(0, len(time_series)), n)

    time_series_w_nan = time_series.copy()
    time_series_w_nan[rand_num] = np.nan







