import time
import numpy as np
import scipy.sparse as sps
from sklearn.decomposition import NMF

np.random.seed(20170430)

""" Create sparse data """

n_user = 500000
n_item = 100000
n_data = 1000000

nnz_i, nnz_j, nnz_val = np.random.choice(n_user, size=n_data), \
                        np.random.choice(n_item, size=n_data), \
                        np.random.random(size=n_data)
X =  sps.csr_matrix((nnz_val, (nnz_i, nnz_j)), shape=(n_user, n_item))
print('X-shape: ', X.shape, ' X nnzs: ', X.nnz)
print('type(X): ', type(X))
# <class 'scipy.sparse.csr.csr_matrix'> #                          !!!!!!!!!!

""" NMF """
model = NMF(n_components=50, init='random', random_state=0)

start_time = time.time()
W = model.fit_transform(X)
end_time = time.time()

print('Used (secs): ', (end_time - start_time))
print(model.reconstruction_err_)
print(model.n_iter_)
print(W)