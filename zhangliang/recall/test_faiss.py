import faiss
import numpy as np
import os
from zhangliang.utils.config import get_ml_data_dir, get_ml_train_path, get_ml_test_path
from zhangliang.utils.dictionary import load_dict


def test_faiss():
	d = 64
	# 向量维度
	nb = 100000
	# 待索引向量size
	nq = 10000
	# 查询向量size
	np.random.seed(1234)
	# 随机种子确定
	xb = np.random.random((nb, d)).astype('float32')
	xb[:, 0] += np.arange(nb) / 1000.
	# 为了使随机产生的向量有较大区别进行人工调整向量
	xq = np.random.random((nq, d)).astype('float32')
	xq[:, 0] += np.arange(nq) / 1000.

	index = faiss.IndexFlatL2(d)
	# 建立索引
	print(index.is_trained)
	# 输出true
	index.add(xb)
	# 索引中添加向量
	print(index.ntotal)

	k = 4
	# 返回每个查询向量的近邻个数
	D, I = index.search(xb[:5], k)
	# 检索check
	print(I)
	print(D)
	D, I = index.search(xq, k)
	# xq检索结果
	print(I[:5])
	# 前五个检索结果展示
	print(I[-5:])


def build_faiss_index(embedding_path=None,
					  faiss_index_path=None,
					  embedding_dim=32,
					  faiss_index_key="IDMap,Flat"):
	# embedding_data: id \t embedding_str
	batch_size = 256
	ids, embeddings = [], []

	#index = faiss.index_factory(embedding_dim, "IDMap,Flat")
	quantizer = faiss.IndexFlatL2(embedding_dim)
	nlist = 50
	code_size = 8
	index = faiss.IndexIVFFlat(quantizer, embedding_dim, nlist)
	index.nprobe = 5

	with open(embedding_path, 'r', encoding='utf-8') as fr:
		for line in fr:
			buf = line[:-1].split('\t')
			if len(buf) != 2:
				continue

			_id = int(buf[0])
			_embedding = np.array(list(map(lambda x: float(x), buf[1].split(','))), dtype=np.float32)

			ids.append(_id)
			embeddings.append(_embedding)

			if len(ids) % batch_size == 0:
				embeddings = np.array(embeddings, dtype=np.float32)
				ids = np.array(ids, dtype=np.int)

				if not index.is_trained:
					index.train(embeddings)
				else:
					index.add_with_ids(embeddings, ids)
				ids, embeddings = [], []

	if len(ids) > 0:
		embeddings = np.array(embeddings, dtype=np.float32)
		ids = np.array(ids, dtype=np.int64)
		index.add_with_ids(embeddings, ids)

	faiss.write_index(index, faiss_index_path)


def get_user_seq_dict(rating_path=None):
	# rating: user_str :: item_str :: rating :: timestamp
	# Output: user_seq_dict: {user: item_str_list}
	user_seq_dict = {}
	with open(rating_path, 'r', encoding='utf-8') as fr:
		for line in fr:
			buf = line[:-1].split("::")
			if len(buf) != 4:
				continue

			user_str = buf[0]
			item_str = buf[1]
			if user_str not in user_seq_dict:
				user_seq_dict[user_str] = [item_str]
			else:
				user_seq_dict[user_str].append(item_str)

	return user_seq_dict


def recall_items_by_faiss(faiss_index_path=None,
						  train_rating_path=None,
						  test_rating_path=None,
						  user_embedding_path=None,
						  item_mapping_dict=None,
						  recall_path=None,
						  k=1):
	# Dirty codes
	# rating: user_str :: item_str :: rating :: timestamp
	# user_embedding_path: user_str \t user_embedding(',')
	# item_mapping_dict: {item_str: item_id}
	#
	# Output: (test_user_str)  true_item_id_list \t pred_item_id_list

	# === Load faiss_index
	faiss_index = faiss.read_index(faiss_index_path)
	print("Load faiss_index done!")

	# === Load user_embedding_dict: {user_str: user_embedding(np.array)}
	# user_embedding: user_str \t embedding
	user_embedding_dict = {}
	with open(user_embedding_path, 'r', encoding='utf-8') as fr:
		for line in fr:
			buf = line[:-1].split('\t')
			if len(buf) != 2:
				continue
			user_str = buf[0]
			user_embedding = np.array(list(map(lambda x: float(x), buf[1].split(','))))
			user_embedding_dict[user_str] = user_embedding
	print("Load user_embedding_dict done! #user_embedding=%d" % len(user_embedding_dict))

	# === Get seq of users: {user_str: behavior_seq} (Order of seq is not needed.)
	train_user_true_seq_dict = get_user_seq_dict(rating_path=train_rating_path)

	# item_str to item_id (for result of faiss is item_id),
	# {user: item_str_list} ==> {user: item_id_list}
	test_user_true_seq_dict = get_user_seq_dict(rating_path=test_rating_path)

	# === Recall predict for each test user.
	fw = open(recall_path, 'w', encoding='utf-8')
	for user_str, test_seq_list in test_user_true_seq_dict.items():
		if user_str not in user_embedding_dict:
			continue

		# === get test_true_id_set
		test_seq_set = set(test_seq_list)
		# item_str to item_id

		test_true_id_list = []
		for item_str in test_seq_set:
			if item_str in item_mapping_dict:
				item_id = item_mapping_dict[item_str]
				test_true_id_list.append(item_id)

		test_true_id_set = set(test_true_id_list)
		if len(test_true_id_set) == 0:
			continue

		# === get test_pred_id_set
		# train_seq_set
		train_id_set = set()
		if user_str in train_user_true_seq_dict:
			train_seq_set = set(train_user_true_seq_dict[user_str])
			#train_id_set = set([item_mapping_dict[item_str] for item_str in train_seq_set])
			train_id_list = []
			for item_str in train_seq_set:
				if item_str in item_mapping_dict:
					item_id = item_mapping_dict[item_str]
					train_id_list.append(item_id)
			train_id_set = set(train_id_list)

		user_embedding = np.array([user_embedding_dict[user_str]], dtype=np.float32) # must be np.float32
		# [1, k] => [k, ]
		test_pred_id_list = recall_by_faiss(faiss_index=faiss_index, k=k, query_vecs=user_embedding)[0]

		# remove train_seq
		test_pred_id_set = set(test_pred_id_list) - train_id_set
		fw.write(set_to_str(test_true_id_set) + '\t' + set_to_str(test_pred_id_set) + '\n')

	fw.close()
	print("Write done! %s" % recall_path)


def set_to_str(_set):
	return ','.join(list(map(lambda x: str(x), _set)))

def recall_by_faiss(faiss_index=None, k=1, query_vecs=None):
	ret = faiss_index.search(query_vecs, k=k)
	distances, neighbor_ids = faiss_index.search(query_vecs, k)
	return neighbor_ids


if __name__ == "__main__":
	#test_faiss()

	user_embedding_path = os.path.join(get_ml_data_dir(), "mf" + "_user_embedding.txt")

	item_embedding_path = os.path.join(get_ml_data_dir(), "mf" + "_item_embedding.txt")
	item_faiss_path = os.path.join(get_ml_data_dir(), "mf_item_faiss.index")

	# ! Ignore validate data.
	train_rating_path = get_ml_train_path()
	test_rating_path = get_ml_test_path()

	recall_path = os.path.join(get_ml_data_dir(), "mf_faiss_recall.text")

	item_mapping_dict_path = os.path.join(get_ml_data_dir(), "item_mapping_dict.pkl")

	recall_at_k = 500

	# === Build faiss index of item_embedding
	faiss_index_key = "IVFx,Flat"
	build_faiss_index(embedding_path=item_embedding_path,
					  faiss_index_path=item_faiss_path,
					  embedding_dim=32,
					  faiss_index_key=faiss_index_key)
	print("Build faiss_index done!", item_faiss_path)


	# === Recall for test_user
	item_mapping_dict = load_dict(item_mapping_dict_path)
	recall_items_by_faiss(faiss_index_path=item_faiss_path,
						  train_rating_path=train_rating_path,
						  test_rating_path=test_rating_path,
						  user_embedding_path=user_embedding_path,
						  item_mapping_dict=item_mapping_dict,
						  recall_path=recall_path,
						  k=recall_at_k)
