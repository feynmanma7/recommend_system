import os
from zhangliang.utils.config import get_ml_data_dir


def recall_at_k(y_true_list, y_pred_list, k=1):
	assert len(y_true_list) == len(y_pred_list)

	total_num_sample = len(y_true_list)
	recall = 0
	for y_true, y_pred in zip(y_true_list, y_pred_list):
		num_intersection = len(set(y_true) & set(y_pred[:k]))
		print(num_intersection, len(y_true))
		recall += num_intersection * 1.0 / len(y_true)

	return recall / total_num_sample


def test_recall():
	y_true_list = [[1, 2, 3, 4, 5], [1, 3, 5, 7, 9]]
	y_pred_list = [[1, 3, 5, 7, 9], [3, 6, 9]]
	k = 1
	ret = recall_at_k(y_true_list, y_pred_list, k=1)
	print(ret)


def recall_on_file(recall_path=None, k=1):
	# recall: true_list \t pred_list; split by ','

	total_recall = 0
	total_num = 0

	with open(recall_path, 'r', encoding='utf-8') as fr:
		for line in fr:
			buf = line[:-1].split('\t')
			if len(buf) != 2:
				continue
			total_num += 1
			true_set = set(buf[0].split(','))
			pred_set = set(buf[1].split(',')[:k])

			intersection = true_set & pred_set
			cur_recall = len(intersection) / len(true_set)
			total_recall += cur_recall

	print('total_num=%d' % total_num)
	total_recall /= total_num
	print('recall@%d=%.4f' % (k, total_recall))


if __name__ == '__main__':
	recall_path = os.path.join(get_ml_data_dir(), "mf_faiss_recall.text")
	k = 500
	recall_on_file(recall_path=recall_path, k=k)

