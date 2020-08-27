from zhangliang.matrix_factorization.mf import MF, test_model_once
from zhangliang.utils.config import get_ml_test_path, \
    get_ml_data_dir, get_log_dir, get_model_dir
from zhangliang.utils.dictionary import load_dict
import tensorflow as tf
tf.random.set_seed(7)
import os, time
import numpy as np
np.random.seed(7)


def test_model(model=None, test_path=None, result_path=None,
               user_mapping_dict=None,
               item_mapping_dict=None):
    # test_data: user :: item :: rating :: tm
    # result: true_rating \t pred_rating

    with open(test_path, 'r', encoding='utf-8') as fr:
        with open(result_path, 'w', encoding='utf-8') as fw:

            line_cnt = 0
            for line in fr:
                buf = line[:-1].split('::')
                user = buf[0]
                item = buf[1]

                if user not in user_mapping_dict or item not in item_mapping_dict:
                    continue

                user_index = np.array([user_mapping_dict[user]], dtype=np.int32)
                item_index = np.array([item_mapping_dict[item]], dtype=np.int32)

                pred_rating = model(inputs=(user_index, item_index))
                pred_rating = pred_rating.numpy()[0]
                #print(pred_rating)

                true_rating = buf[2]

                fw.write(str(true_rating) + '\t' + str(pred_rating) + '\n')
                line_cnt += 1
                if line_cnt % 1000 == 0:
                    print(line_cnt)
            print("Total line %d" % line_cnt)


if __name__ == '__main__':
    method = "mf"
    checkpoint_dir = os.path.join(get_model_dir(), method)
    test_path = get_ml_test_path()
    test_result_path = os.path.join(get_ml_data_dir(), method + "_test_result.txt")

    num_user = 6040
    num_item = 3643
    embedding_dim = 32

    # === Load user, item mapping dict.
    user_mapping_dict_path = os.path.join(get_ml_data_dir(), "user_mapping_dict.pkl")
    item_mapping_dict_path = os.path.join(get_ml_data_dir(), "item_mapping_dict.pkl")

    user_mapping_dict = load_dict(user_mapping_dict_path)
    item_mapping_dict = load_dict(item_mapping_dict_path)

    # === Build and compile model.
    model = MF(num_user=num_user, num_item=num_item, embedding_dim=embedding_dim)
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
    test_model_once(model=model, num_user=num_user, num_item=num_item)

    # === Test
    start = time.time()
    test_model(model=model, test_path=test_path, result_path=test_result_path,
               user_mapping_dict=user_mapping_dict,
               item_mapping_dict=item_mapping_dict)
    end = time.time()
    last = end - start
    print("Write done! %s Lasts %.2fs" % (test_result_path, last))

