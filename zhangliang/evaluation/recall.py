def recall_at_k(y_true_list, y_pred_list, k=1):
	assert len(y_true_list) == len(y_pred_list)

	total_num_sample = len(y_true_list)
	recall = 0
	for y_true, y_pred in zip(y_true_list, y_pred_list):
		num_intersection = len(set(y_true) & set(y_pred[:k]))
		print(num_intersection, len(y_true))
		recall += num_intersection * 1.0 / len(y_true)

	return recall / total_num_sample


if __name__ == '__main__':
	y_true_list = [[1, 2, 3, 4, 5], [1, 3, 5, 7, 9]]
	y_pred_list = [[1, 3, 5, 7, 9], [3, 6, 9]]
	k = 1
	ret = recall_at_k(y_true_list, y_pred_list, k=1)
	print(ret)