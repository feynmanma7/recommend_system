from sklearn.metrics import mean_squared_error
from zhangliang.utils.config import get_ml_data_dir
import os
import numpy as np


def compute_rmse(test_result_path=None):
    # test_result: y_true \t y_pred

    y_true = []
    y_pred = []
    with open(test_result_path, 'r', encoding='utf-8') as fr:
        for line in fr:
            buf = line[:-1].split('\t')
            y_true.append(float(buf[0]))
            y_pred.append(float(buf[1]))

    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    print('mse = %.4f, rmse = %.4f' % (mse, rmse))


if __name__ == '__main__':
    # mf, libfm, fm, deep_fm
    method = "deep_fm"
    test_result_path = os.path.join(get_ml_data_dir(), method + "_test_result.txt")

    compute_rmse(test_result_path=test_result_path)