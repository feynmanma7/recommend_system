# Matrix Factorization

# Data
movielens-1m

Shuffle and split into train:val:test = 3:1:1. 

# Model

> $r_{u, i} = embedding(user) * embedding(item) $

# MF + BatchNormalization

Epoch 13/100
9373/9373 [==============================] - 137s 15ms/step - loss: 0.7013 - acc: 0.0556 - val_loss: 0.7998 - val_acc: 0.0568

Model: "mf"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
embedding (Embedding)        multiple                  193280    
_________________________________________________________________
embedding_1 (Embedding)      multiple                  116576    
_________________________________________________________________
batch_normalization (BatchNo multiple                  128       
_________________________________________________________________
batch_normalization_1 (Batch multiple                  128       
=================================================================
Total params: 310,112
Trainable params: 309,984
Non-trainable params: 128
_________________________________________________________________
None

Train done! Lasts: 2530.26s

# Evaluation

RMSE (Root-Mean-Squared-Error)

embedding_dim = 32

|model|Train|Val|Test|note
|----|-----|---|----|---|
|MF|0.9148|0.9148|0.9194||
|MF|1.4107|0.9705|0.9905|dropout=0.5, epochs=43, not converge yet, 130s per epoch|
|MF|0.8374|0.8943|0.8933|batch_normalization, epoch=13, 137s per epoch|

