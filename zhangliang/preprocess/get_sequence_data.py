from pyspark import SparkContext
from pyspark.sql.types import *
from pyspark.sql import SparkSession
from zhangliang.utils.config import get_ml_data_dir
import os


def _to_int(arr):
    return list(map(lambda x: int(x), arr))


if __name__ == '__main__':
    data_path = os.path.join(get_ml_data_dir(), "ratings.dat")
    sorted_data_path = os.path.join(get_ml_data_dir(), "sorted_rating")

    sc = SparkContext(master="local", appName="movie_lens")
    data = sc.textFile(data_path).map(lambda x: x.split("::")).map(_to_int)
    #print(data.map(lambda x: x.split("::")).take(1))

    schema = StructType([
        StructField("user", IntegerType(), True),
        StructField("movie", IntegerType(), True),
        StructField("rating", IntegerType(), True),
        StructField("timestamp", IntegerType(), True)
    ])

    ss = SparkSession(sc)
    df = ss.createDataFrame(data, schema).sort("timestamp")

    print(df.show(5))

    df.coalesce(1).write.option("header", "false").csv(sorted_data_path)
    print("Write done!", sorted_data_path)



