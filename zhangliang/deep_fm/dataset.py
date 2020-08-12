from zhangliang.utils.config import get_ml_train_path, get_ml_val_path, get_ml_data_dir
from zhangliang.utils.dictionary import load_dict
import tensorflow as tf
import numpy as np
import os


def dataset_generator(data_path=None,
                      epochs=10,
                      shuffle_buffer_size=1024,
                      batch_size=16,
                      user_mapping_dict=None,
                      item_mapping_dict=None):
    # input_data: user::item::rating::tm
    # Output: (input_values, input_indexes, bias_indexes)
    #   input_values: [1, 1], 1 for categorical feature, real-value-itself for numerical feature.
    #   input_indexes: [user_index, item_index], item_index = raw_item_index + num_user for FM
    #   bias_indexes: [num_user + num_item]

    num_user = len(user_mapping_dict)
    num_item = len(item_mapping_dict)
    bias_index = num_user + num_item

    def generator():
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                buf = line[:-1].split('::')

                if buf[0] not in user_mapping_dict or buf[1] not in item_mapping_dict:
                    continue

                user_index = int(user_mapping_dict[buf[0]])
                item_index = int(item_mapping_dict[buf[1]]) + num_user # For FM
                rating = float(buf[2])

                input_values = [1, 1]
                input_indexes = [user_index, item_index]
                bias_indexes = [bias_index]

                inputs = (input_values, input_indexes, bias_indexes)
                y_true = np.array([rating], dtype=np.float32)

                yield inputs, y_true

    input_len = 2 # [user_index, item_index]
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
                user_mapping_dict=None,
                item_mapping_dict=None
                ):
    return dataset_generator(data_path=data_path,
                             epochs=epochs,
                             shuffle_buffer_size=shuffle_buffer_size,
                             batch_size=batch_size,
                             user_mapping_dict=user_mapping_dict,
                             item_mapping_dict=item_mapping_dict)


if __name__ == "__main__":
    train_path = get_ml_train_path()
    user_mapping_dict_path = os.path.join(get_ml_data_dir(), "user_mapping_dict.pkl")
    item_mapping_dict_path = os.path.join(get_ml_data_dir(), "item_mapping_dict.pkl")

    user_mapping_dict = load_dict(user_mapping_dict_path)
    item_mapping_dict = load_dict(item_mapping_dict_path)

    train_dataset = get_dataset(data_path=train_path,
                                batch_size=4,
                                user_mapping_dict=user_mapping_dict,
                                item_mapping_dict=item_mapping_dict
                                )

    # users:[1, ], items: [1, ]
    # rating: [1, ]

    """
    for i, (outputs) in zip(range(2), train_dataset):
        print(outputs)
    """

    for i, ((input_values, input_indexes, bias_indexes), ratings) in zip(range(5), train_dataset):
        print(i, input_values.shape, input_indexes.shape, bias_indexes.shape, ratings.shape)

