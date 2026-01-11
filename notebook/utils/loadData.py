import pandas as pd

def load_data(path):
    return pd.read_csv(path)

def set_up():
    learn_X = load_data('../data/interim/learn_X.csv')
    learn_y = load_data('../data/interim/learn_y.csv')
    test_X = load_data('../data/interim/test_X.csv')
    test_y = load_data('../data/interim/test_y.csv')
    return learn_X, learn_y, test_X, test_y