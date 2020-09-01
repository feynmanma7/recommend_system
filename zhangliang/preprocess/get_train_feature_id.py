from zhangliang.utils.config import get_ml_data_dir
import os, pickle


def load_dict(dict_path):
	with open(dict_path, 'rb') as fr:
		return pickle.load(fr)


def dump_dict(dict_path, dict):
	with open(dict_path, 'wb') as fw:
		print("#dict = %d" % len(dict))
		pickle.dump(dict, fw)


if __name__ == "__main__":
	train_val_path = os.path.join(get_ml_data_dir(), "sorted_train_val.dat")
	user_path = os.path.join(get_ml_data_dir(), "users.dat")
	movie_path = os.path.join(get_ml_data_dir(), "movies.dat")

	user_dict_path = os.path.join(get_ml_data_dir(), "dict", "user.dict")
	movie_dict_path = os.path.join(get_ml_data_dir(), "dict", "movie.dict")

	age_period_dict_path = os.path.join(get_ml_data_dir(), "dict", "age_period.dict")
	occupation_dict_path = os.path.join(get_ml_data_dir(), "dict", "occupations.dict")
	zip_code_dict_path = os.path.join(get_ml_data_dir(), "dict", "zip_code.dict")

	genres_dict_path = os.path.join(get_ml_data_dir(), "dict", "genres.dict")

	user_dict = load_dict(user_dict_path)
	movie_dict = load_dict(movie_dict_path)

	age_period_dict, occupation_dict, zip_code_dict = {}, {}, {}
	age_period_id, occupation_id, zip_code_id = 0, 0, 0

	with open(user_path, 'r', encoding='utf-8') as fr:
		# 1::F::1::10::48067, user_str gender age_period occupation zip_code
		for line in fr:
			buf = line[:-1].split("::")
			if len(buf) != 5:
				continue
			user = buf[0]
			if user not in user_dict:
				continue

			age_period = buf[2]
			occupation = buf[3]
			zip_code = buf[4]

			if age_period not in age_period_dict:
				age_period_dict[age_period] = age_period_id
				age_period_id += 1
			if occupation not in occupation_dict:
				occupation_dict[occupation] = occupation_id
				occupation_id += 1
			if zip_code not in zip_code_dict:
				zip_code_dict[zip_code] = zip_code_id
				zip_code_id += 1

		dump_dict(age_period_dict_path, age_period_dict)
		dump_dict(occupation_dict_path, occupation_dict)
		dump_dict(zip_code_dict_path, zip_code_dict)

