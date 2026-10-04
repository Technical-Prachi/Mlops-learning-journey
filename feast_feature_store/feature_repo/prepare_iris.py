import pandas as pd
from datetime import datetime

df = pd.read_csv("data/iris.csv")

# ID column
df["id"] = range(len(df))

# Species -> Target
mapping = {
    "setosa": 0,
    "versicolor": 1,
    "virginica": 2
}

df["target"] = df["species"].map(mapping)

# Event timestamp
df["event_timestamp"] = datetime.now()

# Save updated dataset
df.to_csv("data/iris_feast.csv", index=False)

print("iris_feast.csv created successfully!")
