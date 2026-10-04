import pandas as pd

def test_dataset_exists():
    df = pd.read_csv("data/iris.csv")
    assert len(df) > 0

def test_no_missing_values():
    df = pd.read_csv("data/iris.csv")
    assert df.isnull().sum().sum() == 0

def test_columns():
    df = pd.read_csv("data/iris.csv")

    expected = [
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width",
        "species",
    ]

    assert list(df.columns) == expected

def test_feature_ranges():
    df = pd.read_csv("data/iris.csv")

    assert df["sepal_length"].between(4,8).all()
    assert df["sepal_width"].between(2,5).all()
