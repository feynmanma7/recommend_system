
# Logistic Regression

> $\log \frac{p(y=1)}{p(y=0)} = \sum_i w_i x_i = W^TX$

> $p(y=1) = \frac{1}{1 + exp(-W^TX)}$

# Matrix Factorization

> $Y = U^TI$, $U \in {m*d}, I \in {n*d}$

> $Y_{pred} = U_{pred}^T I_{pred}$, $U_{pred} \in {m*k}, I_{pred} \in {n*k}$

> $y_{i, j} = <U_i, I_j>=U_i^TI_j$

# Factorization Machine

> $y(x_1,x_2,...,x_n) = w_0 + \sum_{i=1}^n w_i x_i + \sum_{i=1}^n \sum_{j=i+1}^n <v_i, v_j> x_i x_j$ 

> $V \in \mathbb{R}^{n * k}$

> $<v_i, v_j> = \sum_{f=1}^k v_{i, f} * v_{j, f}$

|i/j|1|2|...|n|
|--|--|--|--|--|
|1|\\|*|*|*|
|2|-|\\|*|*|
|...|-|-|\\|*|
|n|-|-|-|\\|

> $\sum_i \sum_j <v_i, v_j>x_i x_j$

> $= 2 * \sum_i \sum_{j=i+1} <v_i, v_j> x_i x_j + \sum_i <v_i, v_i> x_i x_i$

Then, 

> $\sum_i \sum_{j=i+1}<v_i, v_j>x_i x_j$

> $=\frac{1}{2}\sum_i \sum_j <v_i, v_j>x_i x_j - \frac{1}{2} \sum_i <x_i, x_i> x_i^2$

> $=\frac{1}{2} (\sum_i \sum_j \sum_{f=1}^k v_{f, i} v_{f, j}x_i x_j - \sum_i \sum_{f=1}^k v_{i, f}v_{i, f} x_i x_i)$

> $=\frac{1}{2} \sum_{f=1}^k ((\sum_i v_{f, i}x_i)(\sum_j v_{f, j}x_j) - \sum_i v_{i, f}^2 x_i^2)$

> $=\frac{1}{2}\sum_{f=1}^k ((\sum_i v_{f, i})^2 - \sum_i v_{i, f}^2 x_i^2)$

Time complexity turns from $O(n^2k)$ to $O(kn)$.

## LibFM

[http://libfm.org/](http://libfm.org/)

+ Download libfm-1.42.src.tar.gz

> tar -xzf libfm-1.42.src.tar.gz

+ Transform Movielens data to libfm format use libfm-1.42.src/scripts/triple_format_to_libfm.pl

> ./triple_format_to_libfm.pl -in train_rating,val_rating,test_rating -target 2  -delete_column 3 -separator "::"

+ Train libfm

> time ./libFM -task r -train /ml-1m/train_rating.libfm -test /ml-1m/test_rating.libfm -dim '1,1,8' -out /ml-1m/libfm_result.txt

Train metric, 

> \#Iter= 99	Train=0.820756	Test=0.870558

Test result, 

> 4.23288

> 3.1636

> 2.75591

> 2.60068

> 4.78376

+ Evaluation

|Model|Train|Val|Test|
|---|-----|---|----|
|MF|0.9148|0.9148|0.9194|
|LibFM|--|--|0.8706|

## Tensorflow-FM

> $y(x_1, x_2, ..., x_n) = w_0 + \sum_{i=1} w_i x_i + \sum_i \sum_{j=i+1} <v_i, v_j> x_i x_j$

> $= (w_0 + \sum_{i=1} w_i x_i) + \frac{1}{2}\sum_{f=1}^k ((\sum_i v_{f, i})^2 - \sum_i v_{i, f}^2 x_i^2)$

### Train

Epoch 3/100
9373/9373 [==============================] - 157s 17ms/step - loss: 0.7210 - acc: 0.0556 - val_loss: 0.7772 - val_acc: 0.0571

sqrt(0.7210) = 0.8491
sqrt(0.7772) = 0.8816

### Test


## Evaluation

# Field Factorization Machine


> $w_{i, j} = V_{i, f_j}^T V_{j, f_i}$

# DeepFM




# xDeepFM

