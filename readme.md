# Data
Movielens-1m



# Evaluation

RMSE (Root-Mean-Squared-Error)

|Model|Train|Val|Test|Note|
|---|-----|---|----|---|
|MF|0.9148|0.9148|0.9194||
|MF|1.4107|0.9705|0.9905|dropout=0.5, epochs=43, not converge yet, 130s per epoch|
|MF|0.8374|0.8943|0.8933|batch_normalization, epoch=13, 137s per epoch|
|LibFM|--|--|0.8706||
|FM|0.7995|0.8816|0.8805|[user,item]|
|DeepFM|0.8450|0.8964|0.8953|epoch=3|

