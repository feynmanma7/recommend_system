from zhangliang.matrix_factorization.mf import MF, test_model_once
from zhangliang.utils.config import get_ml_test_path, \
	get_ml_data_dir, get_log_dir, get_model_dir
from zhangliang.utils.dictionary import load_dict
import tensorflow as tf
tf.random.set_seed(7)
import os, time
import numpy as np
np.random.seed(7)


def get_user_embedding(model=None, user_embedding_path=None, user_mapping_dict=None):
	batch_size = 64
	users = []
	fw = open(user_embedding_path, 'w')
	for user, user_id in user_mapping_dict.items():
		user.append(user_id)

		if len(users) == batch_size:
			user_embedding = model.get_user_embedding(np.array(users))
			for user_id, embedding in zip(users, user_embedding):
				fw.write(str(user_id) + '' + ,'.join(list(map(lambda x: str(x), embedding))))




if __name__ == '__main__':
	method = "mf"
	checkpoint_dir = os.path.join(get_model_dir(), method)
	user_embedding_path = os.path.join(get_ml_data_dir(), method + "_user_embedding.txt")
	item_embedding_path = os.path.join(get_ml_data_dir(), method + "_item_embedding.txt")

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
	print(model.summary())

	user_embedding = model.get_user_embedding(np.array([1, 2, 3])).numpy()
	for embedding in user_embedding:
		print(','.join(list(map(lambda x: str(x), embedding))))

	# === Get user embedding
	"""
	get_user_embedding(model=model, user_mapping_dict=user_mapping_dict,
					   user_embedding_path=user_embedding_path)
	print("Get user embedding done! %s" % user_embedding_path)
	"""


	# === Get item embedding



