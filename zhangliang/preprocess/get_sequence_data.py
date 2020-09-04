from pyspark import SparkContext
from pyspark.sql.types import *
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
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

"""

def _to_int(arr):
    return list(map(lambda x: int(x), arr))


def _sort_by_tm(movie_tm):
    buf = movie_tm.split(',')
    movie = buf[0::2]
    tm = buf[1::2]
    return ','.join(list(map(lambda x:x[0], sorted(zip(movie, tm), key=lambda x:x[1]))))


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
    rating_path = os.path.join(get_ml_data_dir(), "word2vec", "sorted_train_val.dat")

    spark_context = SparkContext(master="local", appName="movie_lens")
    spark_session = SparkSession(spark_context)
    seq_path = os.path.join(get_ml_data_dir(), "word2vec", "train_val_seq")
    fw = open(seq_path, 'w', encoding='utf-8')

    rating_df = load_rating(spark_context=spark_context,
                            spark_session=spark_session,
                            data_path=rating_path)

    movie_df = rating_df.groupBy('user').agg(F.collect_list(
        F.concat_ws(',', rating_df["movie"], rating_df["timestamp"])
    ).alias("movie_tm"))

    for data in movie_df.collect():
        user = data["user"]
        movie_tm = data["movie_tm"]
        sorted_seq = sorted(list(map(lambda x: x.split(','), movie_tm)), key=lambda x:x[1])
        movie_seq = ','.join(list(map(lambda x: str(x[0]), sorted_seq)))
        fw.write(movie_seq + '\n')
    fw.close()
    print("Write seq done!", seq_path)

    #rating_df.coalesce(1).write.option("header", "false").csv(sorted_data_path)
    #print("Write done!", sorted_data_path)
