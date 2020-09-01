from zhangliang.utils.config import get_ml_data_dir
import os, pickle


def load_dict(dict_path):
	with open(dict_path, 'rb') as fr:
		return pickle.load(fr)


if __name__ == "__main__":
	data_path = os.path.join(get_ml_data_dir(), "sorted_test.dat")
	feature_path = os.path.join(get_ml_data_dir(), "test.dat")

	user_path = os.path.join(get_ml_data_dir(), "users.dat")
	movie_path = os.path.join(get_ml_data_dir(), "movies.dat")

	user_dict_path = os.path.join(get_ml_data_dir(), "dict", "user.dict")
	movie_dict_path = os.path.join(get_ml_data_dir(), "dict", "movie.dict")

	age_period_dict_path = os.path.join(get_ml_data_dir(), "dict", "age_period.dict")
	occupation_dict_path = os.path.join(get_ml_data_dir(), "dict", "occupations.dict")
	zip_code_dict_path = os.path.join(get_ml_data_dir(), "dict", "zip_code.dict")

	user_dict = load_dict(user_dict_path)
	movie_dict = load_dict(movie_dict_path)

	age_period_dict = load_dict(age_period_dict_path)
	occupation_dict = load_dict(occupation_dict_path)
	zip_code_dict = load_dict(zip_code_dict_path)

	gender_dict = {'M': 1, 'F': 0}

	# === Load user_features
	user_feature_dict= {} # {user_id: user_feature_dict}
	with open(user_path, 'r', encoding='utf-8') as fr:
		# 1::F::1::10::48067, user_str gender age_period occupation zip_code
		for line in fr:
			buf = line[:-1].split("::")
			if len(buf) != 5:
				continue
			user = buf[0]

			if user not in user_dict:
				continue

			user_id = user_dict[user]

			gender = buf[1]
			gender_id = gender_dict[gender]

			age_period = buf[2]
			age_period_id = age_period_dict[age_period]

			occupation = buf[3]
			occupation_id = occupation_dict[occupation]

			zip_code = buf[4]
			if zip_code not in zip_code_dict:
				continue
			zip_code_id = zip_code_dict[zip_code]

			user_feature_dict[user_id] = {'gender': gender_id,
							       'age_period': age_period_id,
								   'occupation': occupation_id,
								   'zip_code': zip_code_id}


	num_user = len(user_dict)
	num_movie = len(movie_dict)
	num_gender = len(gender_dict)
	num_age_period = len(age_period_dict)
	num_occupation = len(occupation_dict)
	num_zip_code = len(zip_code_dict)

	#total_index = num_movie + num_gender + num_age_period + num_occupation + num_zip_code
	total_index = num_user + num_movie + num_gender + num_age_period + num_occupation + num_zip_code
	print("Total_index = %d" % total_index)

	fw = open(feature_path, 'w', encoding='utf-8')
	with open(data_path, 'r', encoding='utf-8') as fr:
		# data: user,movie,rating,timestamp
		# feature: user_id, movie_id, gender_id, age_period_id, occupation_id, zip_code_id, rating, timestamp
		for line in fr:
			buf = line[:-1].split('::')
			if len(buf) != 4:
				continue
			user = buf[0]
			if user not in user_dict:
				continue

			movie = buf[1]
			if movie not in movie_dict:
				continue

			rating = buf[2]
			timestamp = buf[3]

			user_id = user_dict[user]
			movie_id = movie_dict[movie]

			user_feature = user_feature_dict[user_id]
			gender_id = user_feature['gender']
			age_period_id = user_feature['age_period']
			occupation_id = user_feature['occupation']
			zip_code_id = user_feature['zip_code']

			# Total_index = num_user + num_movie + ... + num
			user_index = user_id
			movie_index = movie_id + num_user
			gender_index = gender_id + num_user + num_movie
			age_period_index = age_period_id + num_user + num_movie + num_gender
			occupation_index = occupation_id + num_user + num_movie + num_gender + num_age_period
			zip_code_index = zip_code_id + num_user + num_movie + num_gender + num_age_period + num_occupation

			feature_list = [user_id, movie_id, gender_id, age_period_id,
							occupation_id, zip_code_id, rating, timestamp]

			fw.write(','.join(list(map(lambda x:str(x), feature_list))) + '\n')
		fw.close()
		print("Write features done! %s" % feature_path)
