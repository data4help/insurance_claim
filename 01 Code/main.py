
# %% Preliminaries

### Packages

# Basic
import pandas as pd
import numpy as np
from tqdm import tqdm
import math
# Plotting
import matplotlib.pyplot as plt
import seaborn as sns
# Pipelines
from sklearn.pipeline import make_pipeline, make_union
from sklearn.base import TransformerMixin, BaseEstimator
# Missing Values
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from category_encoders import TargetEncoder
# Imbalance
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE

# Paths
MAIN_PATH = r"/Users/PM/Documents/Projects/insurance_claim"
RAW_PATH = rf"{MAIN_PATH}/00 Raw"
CODE_PATH = rf"{MAIN_PATH}/01 Code"
DATA_PATH = rf"{MAIN_PATH}/02 Data"
OUTPUT_PATH = rf"{MAIN_PATH}/03 Output"

# Data
train_data = pd.read_csv(rf"{RAW_PATH}/train.csv")
test_data = pd.read_csv(rf"{RAW_PATH}/test.csv")

### SMALL DATASET for running the data
train_data = train_data.loc[:1_000, :]

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

train_data.replace(-1, np.nan, inplace=True)
total_data_wo_target = train_data.drop(columns="target")
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
train_data.drop(columns=highly_missing_columns, inplace=True)

# %% Allocate variable to correct category


def categorical_check(column, data):
    series = data.loc[:, column].dropna()
    bool_series_cat = series.nunique() / len(series) < 0.05
    bool_binary = list(set(series) - set([0, 1])) == []
    return bool_series_cat and not bool_binary


def binary_check(column, data):
    series = data.loc[:, column].dropna()
    bool_binary = list(set(series) - set([0, 1])) == []
    return bool_binary


def float_check(column, data):
    series = data.loc[:, column].dropna()
    bool_float = series.nunique() / len(series) > 0.05
    return bool_float


y = train_data.loc[:, "target"]
feature_data = train_data.drop(columns=["target", "id"])
all_columns = feature_data.columns
cat_variables = [x for x in all_columns if categorical_check(x, feature_data)]
bin_variables = [x for x in all_columns if binary_check(x, feature_data)]
float_variables = [x for x in all_columns if float_check(x, feature_data)]

assert bool(set(cat_variables) & set(bin_variables) & set(float_variables)) is False, "Multiple Columns in diff Cat"
assert len(cat_variables) + len(bin_variables) + len(float_variables) == len(all_columns), "Not complete"

"""
We find that our definition of categorical variables diverges from the one specified already.
This is that we find variables which are defined as a float, but have more than 1 value in
each variable. Given that this variable is therefore not continuous but rather discrete,
we can also treat it like that.
"""


def cat_variables_adjust(data, column):
    """
    This function checks whether a function is already an integer variable, if that is not the
    case then it is changed to one. This is necessary given that integer variables with
    nans are incorrectly changed to a float. Also variables which have float values but
    are actually categorical are changed through that function
    """
    series = data.loc[:, column]
    if series.dtype == int:
        return series.astype(object)
    else:
        num_bins = len(series.value_counts().index)
        cut_series = pd.cut(series , bins=num_bins, labels=list(range(num_bins)))
        return cut_series.astype(object)


for column in cat_variables + bin_variables:
    feature_data.loc[:, column] = cat_variables_adjust(feature_data, column)

# %% Imputation of missing data

"""
Now we impute the missing data, this is done by first mean encoding all categorical
variables. In the second instance we then apply the MICE algorithm which fits
a linear regression onto the missing columns. The entirety is wrapped into a pipeline.
"""


class ColumnSelector(BaseEstimator, TransformerMixin):
    """This class is necessary in order to select the different column groups"""
    def __init__(self, columns):
        self.columns = columns

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        return x[self.columns]


processing_pipeline = make_pipeline(
    # If using make_union, then we HAVE to first select all the columns we will pull from.
    ColumnSelector(all_columns),
    make_union(
        # We start by 'holding out' the binary and float variables
        make_pipeline(ColumnSelector(bin_variables),
                      ),
        make_pipeline(ColumnSelector(float_variables)
                      ),
        # The pipeline for the categorical variables includes mean encoding
        make_pipeline(
            ColumnSelector(cat_variables),
            TargetEncoder(handle_missing="return_nan")
        )
    ),
    IterativeImputer()
)



def test_processing_pipeline():
    processed = processing_pipeline.fit_transform(feature_data, y)
    new_column_order = bin_variables + float_variables + cat_variables
    df_processed = pd.DataFrame(processed, columns=new_column_order)

    # Checking whether we still face nans
    assert not df_processed.isna().any().any(), "There are still missing values"


test_processing_pipeline()

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




# %% Model building





# %%



