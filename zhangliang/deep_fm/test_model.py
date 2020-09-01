from zhangliang.deep_fm.deep_fm import DeepFM, test_model_once
from zhangliang.utils.config import get_ml_test_path, \
    get_ml_data_dir, get_log_dir, get_model_dir
from zhangliang.utils.dictionary import load_dict
import tensorflow as tf
tf.random.set_seed(7)
import os, time
import numpy as np
np.random.seed(7)


def test_model(model=None,
               test_path=None,
               result_path=None,
               total_index=11480):
    # result: true_rating \t pred_rating

    with open(test_path, 'r', encoding='utf-8') as fr:
        with open(result_path, 'w', encoding='utf-8') as fw:

            line_cnt = 0

            batch_size = 128
            input_indexes = []
            true_ratings = []

            for line in fr:
                buf = line[:-1].split(',')
                if len(buf) != 8:
                    continue

                line_cnt += 1
                if line_cnt % 1000 == 0:
                    print(line_cnt)

                user_index = int(buf[0])
                movie_index = int(buf[1])
                gender_index = int(buf[2])
                age_period_index = int(buf[3])
                occupation_index = int(buf[4])
                zip_code_index = int(buf[5])
                true_rating = float(buf[6])

                true_ratings.append(true_rating)

                input_indexes.append([user_index, movie_index, gender_index,
                                           age_period_index, occupation_index, zip_code_index])

                """
                input_values = np.array([[1, 1, 1, 1, 1, 1]], dtype=np.float32)
                input_indexes = np.array([[user_index, movie_index, gender_index,
                                           age_period_index, occupation_index, zip_code_index]],
                                         dtype=np.int32)
                bias_indexes = np.array([[total_index]], dtype=np.int32)
                """

                if line_cnt % batch_size == 0:
                    # [batch_size, 6]
                    input_values = np.array([[1] * 6 for _ in range(batch_size)], dtype=np.float32)
                    input_indexes = np.array(input_indexes, dtype=np.int32)
                    bias_indexes = np.array([[total_index] for _ in range(batch_size)], dtype=np.int32)

                    inputs = (input_values, input_indexes, bias_indexes)

                    # [batch_size, 1]
                    pred_ratings = model(inputs)
                    for true_rating, pred_rating in zip(true_ratings, pred_ratings.numpy()):
                        fw.write(str(true_rating) + '\t' + str(pred_rating[0]) + '\n')

                    true_ratings = []
                    input_indexes = []

            if len(true_ratings) > 0:
                input_values = np.array([[1] * 6 for _ in range(len(input_indexes))], dtype=np.float32)
                bias_indexes = np.array([[total_index] for _ in range(len(input_indexes))], dtype=np.int32)
                input_indexes = np.array(input_indexes, dtype=np.int32)

                inputs = (input_values, input_indexes, bias_indexes)
                pred_ratings = model(inputs)
                for true_rating, pred_rating in zip(true_ratings, pred_ratings.numpy()[0]):
                    fw.write(str(true_rating) + '\t' + str(pred_rating) + '\n')

            print("Total line %d" % line_cnt)


if __name__ == '__main__':
    method = "deep_fm"
    checkpoint_dir = os.path.join(get_model_dir(), method)
    #test_path = get_ml_test_path()
    test_path = os.path.join(get_ml_data_dir(), "test.dat")
    test_result_path = os.path.join(get_ml_data_dir(), method + "_test_result.txt")

    embedding_dim = 8
    dense_units = 32
    dropout_keep_ratio = 0.5

    input_len = 6
    input_dim = 13186

    # === Build and compile model.
    model = DeepFM(input_dim=input_dim,
                   embedding_dim=embedding_dim,
                   dense_units=dense_units,
                   dropout_keep_ratio=dropout_keep_ratio)

    #model = DeepFM(input_len=input_len, input_dim=input_dim, embedding_dim=embedding_dim)
    optimizer = tf.keras.optimizers.Adam(0.001)
    #loss = tf.keras.losses.SparseCategoricalCrossentropy()
    loss = tf.keras.losses.MSE

    model.compile(optimizer=optimizer,
                   loss=loss,
                   metrics=['acc'])

    # === Load weights.
    checkpoint = tf.train.latest_checkpoint(checkpoint_dir=checkpoint_dir)
    model.load_weights(checkpoint)

    # === Run once, to load weights of checkpoint.
    test_model_once(model=model, input_len=input_len, input_dim=input_dim)

    # === Test
    start = time.time()
    test_model(model=model, test_path=test_path, result_path=test_result_path,
               total_index=input_dim)

    end = time.time()
    last = end - start
    print("Write done! %s Lasts %.2fs" % (test_result_path, last))

