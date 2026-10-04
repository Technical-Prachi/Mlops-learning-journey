from datetime import timedelta

from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float32, Int64

iris = Entity(
    name="iris",
    join_keys=["id"],
)

iris_source = FileSource(
       path="data/iris_feast.parquet",
       timestamp_field="event_timestamp",
)

iris_fv = FeatureView(
    name="iris_features",
    entities=[iris],
    ttl=timedelta(days=365),
    schema=[
        Field(name="sepal_length", dtype=Float32),
        Field(name="sepal_width", dtype=Float32),
        Field(name="petal_length", dtype=Float32),
        Field(name="petal_width", dtype=Float32),
        Field(name="target", dtype=Int64),
    ],
    source=iris_source,
)
