from zhangliang.deep_fm.deep_fm import DeepFM
from zhangliang.utils.config import get_ml_train_path, get_ml_val_path, \
    get_ml_data_dir, get_log_dir, get_model_dir
from zhangliang.utils.dictionary import load_dict
from zhangliang.deep_fm.dataset import get_dataset
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
import time
import os


def train_model():
    embedding_dim = 32
    total_num_train = 600209 # num_lines of train.dat
    total_num_val = 200000 # num_lines of val.dat

    total_index = 13186

    dense_units = 32
    dropout_keep_ratio = 0.5

    input_len = 5 # remove user_index

    train_path = os.path.join(get_ml_data_dir(), "train.dat")
    val_path = os.path.join(get_ml_data_dir(), "val.dat")

    model_name = "deep_fm"

    log_dir = os.path.join(get_log_dir(), model_name)
    checkpoint_path = os.path.join(get_model_dir(), model_name, "ckpt")
    history_path = os.path.join(get_log_dir(), "history", model_name + ".pkl")

    epochs = 100
    #epochs = 3
    shuffle_buffer_size = 1024 * 8
    batch_size = 1024
    patience = 6

    # === Load user, item mapping dict.
    num_train_batch = total_num_train // batch_size + 1
    num_val_batch = total_num_val // batch_size + 1

    # === tf.data.Dataset
    train_dataset = get_dataset(data_path=train_path,
                                epochs=epochs,
                                shuffle_buffer_size=shuffle_buffer_size,
                                batch_size=batch_size,
                                total_index=total_index)

    val_dataset = get_dataset(data_path=val_path,
                              epochs=epochs,
                              shuffle_buffer_size=shuffle_buffer_size,
                              batch_size=batch_size,
                              total_index=total_index)

    # === model
    model = DeepFM(input_len=input_len,
            input_dim=total_index,
            embedding_dim=embedding_dim,
            dense_units=dense_units,
            dropout_keep_ratio=dropout_keep_ratio)

    # optimizer
    #optimizer = tf.keras.optimizers.Adam(1e-3, decay=1e-4)
    optimizer = tf.keras.optimizers.Adam(1e-3)

    # loss
    #loss = tf.keras.losses.SparseCategoricalCrossentropy()
    loss = tf.keras.losses.MSE

    model.compile(optimizer=optimizer,
                   loss=loss,
                   metrics=['acc'])

    # callbacks
    callbacks = []

    early_stopping_cb = EarlyStopping(monitor='val_loss',
                                      patience=patience,
                                      restore_best_weights=True)
    callbacks.append(early_stopping_cb)

    tensorboard_cb = TensorBoard(log_dir=log_dir)
    callbacks.append(tensorboard_cb)

    checkpoint_cb = ModelCheckpoint(filepath=checkpoint_path,
                                    save_weights_only=True,
                                    save_best_only=True)
    callbacks.append(checkpoint_cb)

    # === Train
    history = model.fit(train_dataset,
               epochs=epochs,
               steps_per_epoch=num_train_batch,
               validation_data=val_dataset,
               validation_steps=num_val_batch,
               callbacks=callbacks)

    print(model.summary())

    return history


if __name__ == "__main__":
    start = time.time()
    train_model()
    end = time.time()
    last = end - start
    print("\nTrain done! Lasts: %.2fs" % last)