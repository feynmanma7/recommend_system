# Matrix Factorization

# Data
movielens-1m

Shuffle and split into train:val:test = 3:1:1. 

# Model

> $r_{u, i} = embedding(user) * embedding(item) $

# Evaluation

RMSE (Root-Mean-Squared-Error)

embedding_dim = 32

|Train|Val|Test|
|-----|---|----|
|0.9148|0.9148|0.9194|
