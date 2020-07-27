#encoding:utf-8
import numpy as np
np.random.seed(20170430)

def generate_hash_func(num_hash_func=64, prime=23, dim=1000000):
	#hash_funcs = []

	coefs = []

	for i in range(num_hash_func):
		a = np.random.randint(1, 10)
		b = np.random.randint(5, 10)
		#hash_func = lambda x : int(((a * int(x) + b) % prime) % dim)
		#hash_funcs.append(hash_func)

		coefs.append((a, b))

	return coefs
	#return hash_funcs


def minHash(arr, hash_funcs, prime=23, dim=1000000):
	arr = list(map(lambda x:float(x), arr))

	hash_codes = []
	for coef in hash_funcs:
		a, b = coef
		hash_value = list(map(lambda x :
			 int(((a * int(x) + b) % prime) % dim), arr))
		hash_code = min(hash_value)
		hash_codes.append(str(hash_code))

	return ' '.join(hash_codes)


def main(input_path, output_path, prime=23, dim=1000000):
	hash_funcs = generate_hash_func()

	fw = open(output_path, 'w')
	for line in open(input_path, 'r'):
		buf = line[:-1].split('\t')
		item = buf[0]
		vector = buf[1]
		fw.write(item + '\t' + minHash(vector.split(','), hash_funcs) + '\n')

	fw.close()


if __name__ == '__main__':
	input_path = '../data/item_vector.txt'
	output_path = '../data/item_min_hash.txt'
	main(input_path, output_path)
	