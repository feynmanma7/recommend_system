# Train
## Vanilla DeepFM
batch_size = 1024

shuffle_size = 1024 * 8

patience = 10

Epoch 13/100
586/586 [==============================] - 128s 218ms/step - loss: 0.6638 - acc: 0.0556 - val_loss: 0.7800 - val_acc: 0.0570

## DeepFM with BatchNormalization
batch_size = 64

patience = 5

Add BatchNormalization after embedding and dense layers.


Epoch 3/100
9373/9373 [==============================] - 191s 20ms/step - loss: 0.7140 - acc: 0.0555 - val_loss: 0.8035 - val_acc: 0.0570

# Performance

|Model|Train|Val|Test|Note|
|---|-----|---|----|---|
|DeepFM|0.7719|0.8760|-|batch_size=1024, epoch=3, patience=10, test run on FM instead of DeepFM|
|DeepFM|1.2081|0.9235|-|epoch=1, not converge|
|DeepFM|0.8450|0.8964|0.8953|epoch=3|

