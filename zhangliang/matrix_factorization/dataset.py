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
    # Output: (user_index, item_index), rating

    def generator():
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                buf = line[:-1].split('::')

                if buf[0] not in user_mapping_dict or buf[1] not in item_mapping_dict:
                    continue

                user = user_mapping_dict[buf[0]]
                item = item_mapping_dict[buf[1]]
                rating = float(buf[2])

                user = np.array([user], dtype=np.int32)
                item = np.array([item], dtype=np.int32)
                rating = np.array([rating], dtype=np.float32)

                yield (user, item), rating

    dataset = tf.data.Dataset.from_generator(generator,
                                             output_shapes=(
                                                 ((1, ), (1, )),
                                                 (1, )
                                             ),
                                             output_types=(
                                                 (tf.int32, tf.int32),
                                                 tf.float32
                                             ))

    return dataset.repeat(epochs)\
        .shuffle(buffer_size=shuffle_buffer_size)\
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

    for i, ((users, items), ratings) in zip(range(5), train_dataset):
        print(i, users.shape, items.shape, ratings.shape)
