from zhangliang.utils.config import get_ml_train_path, get_ml_data_dir
import os, pickle


def get_mapping_dict(train_path=None,
                     user_mapping_dict_path=None,
                     item_mapping_dict_path=None):
    # train_data: user_id::item_id::rating::tm, 1632::952::2::974718240
    # user_mapping_dict, {user: user_index}
    # item_mapping_dict, {item: item_index}

    user_mapping_dict = {}
    item_mapping_dict = {}
    user_index = 0
    item_index = 0
    with open(train_path, 'r', encoding='utf-8') as fr:
        for line in fr:
            buf = line[:-1].split('::')
            user = buf[0]
            item = buf[1]
            if user not in user_mapping_dict:
                user_mapping_dict[user] = user_index
                user_index += 1
            if item not in item_mapping_dict:
                item_mapping_dict[item] = item_index
                item_index += 1

    print("#user_mapping_dict = %d" % len(user_mapping_dict))
    print("#item_mapping_dict = %d" % len(item_mapping_dict))

    with open(user_mapping_dict_path, 'wb') as fw:
        pickle.dump(user_mapping_dict, fw)

    with open(item_mapping_dict_path, 'wb') as fw:
        pickle.dump(item_mapping_dict, fw)


if __name__ == '__main__':
    train_path = get_ml_train_path()
    user_mapping_dict_path = os.path.join(get_ml_data_dir(), "user_mapping_dict.pkl")
    item_mapping_dict_path = os.path.join(get_ml_data_dir(), "item_mapping_dict.pkl")

    get_mapping_dict(train_path=train_path,
                     user_mapping_dict_path=user_mapping_dict_path,
                     item_mapping_dict_path=item_mapping_dict_path)