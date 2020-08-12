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
    num_user = 6040
    num_item = 3643
    embedding_dim = 32
    total_num_train = 599849 # num_lines of train_rating
    total_num_val = 200113 # num_lines of val_rating

    dense_units = 32
    dropout_keep_ratio = 0.5

    input_len = 2 # [user_index, item_index]
    input_dim = num_user + num_item

    train_path = get_ml_train_path()
    val_path = get_ml_val_path()

    model_name = "deep_fm"

    log_dir = os.path.join(get_log_dir(), model_name)
    checkpoint_path = os.path.join(get_model_dir(), model_name, "ckpt")
    history_path = os.path.join(get_log_dir(), "history", model_name + ".pkl")

    epochs = 100
    #epochs = 3
    shuffle_buffer_size = 1024 * 8
    batch_size = 64 # 1024 --> 64
    patience = 5 # for early stopping

    # === Load user, item mapping dict.
    user_mapping_dict_path = os.path.join(get_ml_data_dir(), "user_mapping_dict.pkl")
    item_mapping_dict_path = os.path.join(get_ml_data_dir(), "item_mapping_dict.pkl")

    user_mapping_dict = load_dict(user_mapping_dict_path)
    item_mapping_dict = load_dict(item_mapping_dict_path)

    num_train_batch = total_num_train // batch_size + 1
    num_val_batch = total_num_val // batch_size + 1

    #print('num_train_batch = %d, num_val_batch = %d' % (num_train_batch, num_val_batch))

    # === tf.data.Dataset
    train_dataset = get_dataset(data_path=train_path,
                                epochs=epochs,
                                shuffle_buffer_size=shuffle_buffer_size,
                                batch_size=batch_size,
                                user_mapping_dict=user_mapping_dict,
                                item_mapping_dict=item_mapping_dict)

    val_dataset = get_dataset(data_path=val_path,
                              epochs=epochs,
                              shuffle_buffer_size=shuffle_buffer_size,
                              batch_size=batch_size,
                              user_mapping_dict=user_mapping_dict,
                              item_mapping_dict=item_mapping_dict)

    # === model
    model = DeepFM(input_len=input_len,
            input_dim=input_dim,
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