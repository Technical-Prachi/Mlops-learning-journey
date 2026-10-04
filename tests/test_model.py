import joblib
import pandas as pd

from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder


def test_model_accuracy():

    model = joblib.load("artifacts/model.pkl")

    df = pd.read_csv("data/iris.csv")

    X = df.iloc[:, :-1]

    y = df.iloc[:, -1]

    encoder = LabelEncoder()
    y = encoder.fit_transform(y)

    pred = model.predict(X)

    acc = accuracy_score(y, pred)

    print("Accuracy:", acc)

    assert acc >= 0.90
