import os


def get_home_dir():
    cwd = os.getcwd()
    return '/'.join(cwd.split('/')[:3])


def get_train_data_dir():
    return os.path.join(get_home_dir(), "data", "recommend_system")


def get_ml_data_dir():
    return os.path.join(get_train_data_dir(), "ml-1m")


def get_ml_train_path():
    return os.path.join(get_ml_data_dir(), "train_rating")


def get_ml_val_path():
    return os.path.join(get_ml_data_dir(), "val_rating")


def get_ml_test_path():
    return os.path.join(get_ml_data_dir(), "test_rating")


def get_log_dir():
    return os.path.join(get_home_dir(), "logs")


def get_model_dir():
    return os.path.join(get_home_dir(), "models")