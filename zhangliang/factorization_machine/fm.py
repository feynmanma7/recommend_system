import tensorflow as tf
tf.random.set_seed(7)
from tensorflow.keras.layers import Input, Dense, Embedding
from tensorflow.keras.models import Model


class FM(tf.keras.Model):
    def __init__(self):
        super(FM, self).__init__()

    def call(self, inputs, training=None, mask=None):
        pass


def build_model(categorical_cols=None,
                integer_cols=None):
    sparse_inputs = [Input(shape=(1, ), dtype=tf.int32, name=col) for col in categorical_cols]
    dense_inputs = [Input(shape=(1, ), dtype=tf.int32, name=col) for col in integer_cols]

    one_order_outputs = OneOrder()([sparse_inputs, dense_inputs])
    embeddings = EmbeddingLayer()([sparse_inputs, dense_inputs])
    two_order_outputs = TwoOrder()(embeddings)

    outputs = one_order_outputs + two_order_outputs
    model = Model(inputs=[sparse_inputs, dense_inputs], outputs=outputs)
    return model



def test_model_once(model=None):
    batch_size = 2
    inputs = tf.random
    outputs =

if __name__ == '__main__':
    model = FM()
    test_model_once(model=model)
