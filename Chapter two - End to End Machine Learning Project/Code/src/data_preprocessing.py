import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import sys
sys.path.append("..")
from scripts.savefig import save_fig







def load_data():
    return pd.read_csv("../data/raw/housing/housing.csv")

def histogram():
    housing = load_data()
    housing.info()
    #print(raw.head())
    print(housing.describe())

    plt.rc('font', size=14)
    plt.rc('axes', labelsize=14, titlesize=14)
    plt.rc('legend', fontsize=14)
    plt.rc('xtick', labelsize=10)
    plt.rc('ytick', labelsize=10)

    housing.hist(bins=50, figsize=(12, 8))
    save_fig("attribute_histogram_plots")  # extra code
    plt.show()

def split_data():
    housing = load_data()
    housing['income_cat'] = pd.cut(housing["median_income"],
                                   bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                                   labels=[1,2,3,4,5])
    strat_train_set, strat_test_set = train_test_split(
        housing, test_size=0.2, stratify=housing["income_cat"], random_state=42)

    for col in (strat_train_set, strat_test_set):
        col.drop("income_cat", axis=1, inplace=True)

    return strat_train_set, strat_test_set

def prep_data():
    trainData, testData= split_data()

    housing = trainData.drop("median_house_value", axis=1)
    housing_labels = trainData["median_house_value"].copy()

    housing_test  = trainData.drop("median_house_value", axis=1)
    housing_test_labels = housing_labels = trainData["median_house_value"].copy()

    return housing, housing_labels
