from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

from sklearn.pipeline import make_pipeline
from pipeline import preprocessed_pipline


def make_LinearRegression():
    lin_reg = make_pipeline(preprocessed_pipline(), LinearRegression())
    # lin_reg.fit(example, label)
    return lin_reg

def malke_DecisionTreeRegressor():
    tree_reg = make_pipeline(preprocessed_pipline(), DecisionTreeRegressor(random_state=42))
    # tree_reg.fit(example, label)
    return tree_reg

if __name__ == '__main__':
    make_LinearRegression()
