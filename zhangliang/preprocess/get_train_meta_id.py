from zhangliang.utils.config import get_ml_data_dir
import os, pickle


def get_train_meta_id(train_val_path=None,
					  user_dict_path=None,
					  movie_dict_path=None):
	# train_val_path: user,movie,rating,timestamp
	# dict: {str: id}, from 0
	user_dict, movie_dict = {}, {}
	user_id, movie_id = 0, 0

	with open(train_val_path, 'r', encoding='utf-8') as fr:
		for line in fr:
			buf = line[:-1].split('::')
			if len(buf) != 4:
				continue
			user = buf[0]
			if user not in user_dict:
				user_dict[user] = user_id
				user_id += 1

			movie = buf[1]
			if movie not in movie_dict:
				movie_dict[movie] = movie_id
				movie_id += 1

		with open(user_dict_path, 'wb') as fw:
			print("#user_dict = %d" % len(user_dict))
			pickle.dump(user_dict, fw)

		with open(movie_dict_path, 'wb') as fw:
			print("#movie_dict = %d" % len(movie_dict))
			pickle.dump(movie_dict, fw)


if __name__ == "__main__":
	train_val_path = os.path.join(get_ml_data_dir(), "sorted_train_val.dat")
	user_dict_path = os.path.join(get_ml_data_dir(), "dict", "user.dict")
	movie_dict_path = os.path.join(get_ml_data_dir(), "dict", "movie.dict")

	get_train_meta_id(train_val_path=train_val_path,
					  user_dict_path=user_dict_path,
					  movie_dict_path=movie_dict_path)

