import pickle


def load_dict(dict_path=None):
    with open(dict_path, 'rb') as fr:
        my_dict = pickle.load(fr)
        return my_dict
    return None