from pyspark import SparkContext
from pyspark.sql.types import *
from pyspark.sql import SparkSession
from zhangliang.utils.config import get_ml_data_dir
import os

"""
# Input
Use ratings.dat, users.dat, movies.dat
+ ratings: user_str, movie_id, rating, timestamp
+ users: user_str, gender, age_period, occupation, zip-code
+ movies: movie_str, title, genres

# Output

ratings order by timestamp

====

To produce  

user_id, movie_id, rating, timestamp
+ user features
gender_id, age_period_id, occupation_id, zip_code_id

+ movie features
genres_id
TODO: title  

"""

def _to_int(arr):
    return list(map(lambda x: int(x), arr))


def load_rating(spark_context=None, spark_session=None, data_path=None):

    def _get_column(line):
        buf = line.split("::")
        user = buf[0]
        item = buf[1]
        rating = float(buf[2])
        timestamp = int(buf[3])
        return [user, item, rating, timestamp]

    schema = StructType([
        StructField("user", StringType(), True),
        StructField("movie", StringType(), True),
        StructField("rating", FloatType(), True),
        StructField("timestamp", IntegerType(), True)
    ])
    data = spark_context.textFile(data_path).map(_get_column)
    df = spark_session.createDataFrame(data, schema)
    return df

if __name__ == '__main__':
    rating_path = os.path.join(get_ml_data_dir(), "ratings.dat")
    user_path = os.path.join(get_ml_data_dir(), "users.dat")
    movie_path = os.path.join(get_ml_data_dir(), "movies.dat")

    spark_context = SparkContext(master="local", appName="movie_lens")
    spark_session = SparkSession(spark_context)
    sorted_data_path = os.path.join(get_ml_data_dir(), "sorted_rating")

    rating_df = load_rating(spark_context=spark_context,
                            spark_session=spark_session,
                            data_path=rating_path)
    print(rating_df.show(5))

    rating_df.coalesce(1).write.option("header", "false").csv(sorted_data_path)
    print("Write done!", sorted_data_path)



