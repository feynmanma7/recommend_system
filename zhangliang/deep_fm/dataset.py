from zhangliang.utils.config import get_ml_train_path, get_ml_val_path, get_ml_data_dir
from zhangliang.utils.dictionary import load_dict
import tensorflow as tf
import numpy as np
import os


def dataset_generator(data_path=None,
                      epochs=10,
                      shuffle_buffer_size=1024,
                      batch_size=16,
                      total_index=11480):

    """
    input_data: user_id, movie_id, gender_id, age_period_id, occupation_id,
                zip_code_id, rating, timestamp

    Output: (input_values, input_indexes, bias_indexes)
       input_values: [6, 1]
            1 for categorical feature, real-value-itself for numerical feature.

       input_indexes: [user_id, movie_id, gender_id, age_period_id, occupation_id,
                zip_code_id, rating]
       bias_indexes: [total_index], bias is in the end of embedding
    """

    def generator():
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                buf = line[:-1].split(',')
                if len(buf) != 8:
                    continue

                user_index = int(buf[0])
                movie_index = int(buf[1])
                gender_index = int(buf[2])
                age_period_index = int(buf[3])
                occupation_index = int(buf[4])
                zip_code_index = int(buf[5])
                rating = float(buf[6])
                #timestamp = buf[7]

                input_values = [1.] * 6
                input_indexes = [user_index, movie_index, gender_index,
                                 age_period_index, occupation_index, zip_code_index]
                bias_indexes = [total_index]

                inputs = (input_values, input_indexes, bias_indexes)
                y_true = np.array([rating], dtype=np.float32)

                yield inputs, y_true

    input_len = 6
    dataset = tf.data.Dataset.from_generator(generator,
                                             output_shapes=(
                                                 ((input_len, ), (input_len, ), (1, )),
                                                 (1, )
                                             ),
                                             output_types=(
                                                 (tf.float32, tf.int32, tf.int32),
                                                 tf.float32
                                             ))

    """
    return dataset.repeat(epochs)\
        .shuffle(buffer_size=shuffle_buffer_size)\
        .batch(batch_size=batch_size)
    """

    # === Shuffle first, then repeat, the whole dataset can be saw in one epoch
    return dataset.shuffle(buffer_size=shuffle_buffer_size) \
        .repeat(epochs) \
        .batch(batch_size=batch_size)


def get_dataset(data_path=None,
                epochs=10,
                shuffle_buffer_size=1024,
                batch_size=16,
                total_index=11480):
    return dataset_generator(data_path=data_path,
                             epochs=epochs,
                             shuffle_buffer_size=shuffle_buffer_size,
                             batch_size=batch_size,
                             total_index=total_index)


if __name__ == "__main__":
    #train_path = get_ml_train_path()
    train_path = os.path.join(get_ml_data_dir(), "train.dat")

    total_index = 13186

    train_dataset = get_dataset(data_path=train_path,
                                batch_size=4,
                                total_index=total_index)

    for i, ((input_values, input_indexes, bias_indexes), ratings) in zip(range(5), train_dataset):
        print(i, input_values.shape, input_indexes.shape, bias_indexes.shape, ratings.shape)

