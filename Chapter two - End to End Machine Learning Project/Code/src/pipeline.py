import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import OneHotEncoder
import data_preprocessing
import sys
sys.path.append("..")
from scripts.cluster_similarity import ClusterSimilarity

def column_ratio(X):
    return X[:, [0]] / X[:, [1]]

def ratio_name(function_transformer, feature_names_in):
    return ["ratio"]  # feature names out

def ratio_pipeline():
    return make_pipeline(
        SimpleImputer(strategy="median"),
        FunctionTransformer(column_ratio, feature_names_out=ratio_name),
        StandardScaler())
def log_pipeline():
    return make_pipeline(
        SimpleImputer(strategy="median"),
        FunctionTransformer(np.log, feature_names_out="one-to-one"),
        StandardScaler())
def cluster_similar():
    return ClusterSimilarity(n_clusters=10, gamma=1., random_state=42)

def default_num_pipeline():
    return make_pipeline(SimpleImputer(strategy="median"),
                                     StandardScaler())
def cat_pipeline():
    return make_pipeline(
    SimpleImputer(strategy="most_frequent"),
    OneHotEncoder(handle_unknown="ignore"))

def preprocessed_pipline():
    return ColumnTransformer([
        ("bedrooms", ratio_pipeline(), ["total_bedrooms", "total_rooms"]),
        ("rooms_per_house", ratio_pipeline(), ["total_rooms", "households"]),
        ("people_per_house", ratio_pipeline(), ["population", "households"]),
        ("log", log_pipeline(), ["total_bedrooms", "total_rooms", "population",
                               "households", "median_income"]),
        ("geo", cluster_similar(), ["latitude", "longitude"]),
        ("cat", cat_pipeline(), make_column_selector(dtype_include=object)),
    ],
    remainder=default_num_pipeline())  # one column remaining: housing_median_age






if __name__ == '__main__':
    housing, housing_label = data_preprocessing.prep_data()
    preprocessed_pipline = preprocessed_pipline()
    housing_prepared = preprocessed_pipline.fit_transform(housing)
    print(housing_prepared.shape)
    print(preprocessed_pipline.get_feature_names_out())