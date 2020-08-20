import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model

def build_model():
    batch_size = 2
    input_dim = 4
    dense_units = 3
    inputs = Input(shape=(input_dim, ))
    outputs = Dense(units=dense_units)(inputs)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.SGD(),
                  loss=tf.keras.losses.BinaryCrossentropy())
    return model


if __name__ == '__main__':
    model = build_model()
    print(model.summary())