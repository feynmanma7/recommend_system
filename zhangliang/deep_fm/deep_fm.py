import tensorflow as tf
tf.random.set_seed(7)
from tensorflow.keras.layers import Dense, Embedding, Dropout, Flatten


class DeepFM(tf.keras.Model):
    def __init__(self,
                 input_len=2,
                 input_dim=10,
                 embedding_dim=8,
                 dense_units=8,
                 dropout_keep_ratio=0.8):
        super(DeepFM, self).__init__()

        self.input_len = input_len
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim

        # === y_fm
        # one-order weights, bias_weight_index = input_dim
        self.one_order_embedding_layer = Embedding(input_dim=input_dim+1, output_dim=1)

        # two-orders weights
        self.two_order_embedding_layer = Embedding(input_dim=input_dim, output_dim=embedding_dim)

        # === y_dnn
        self.dropout1_layer = Dropout(rate=1 - dropout_keep_ratio)
        self.dense_layer = Dense(units=dense_units, activation='relu',
                                 kernel_regularizer=tf.keras.regularizers.l2(0.0001))
        self.dropout2_layer = Dropout(rate=1 - dropout_keep_ratio)
        self.flatten_layer = Flatten()
        self.dnn_output_layer = Dense(units=1, activation='relu')

        # === merge
        self.output_layer = Dense(units=1) # for regression question, activation=linear

    def _compute_fm_1d(self,
                       bias_indexes=None,
                       one_order_embedding=None,
                       input_values=None):
        # bias_indexes: [None, 1]
        # one_order_embedding: [None, input_len, 1]
        # input_values: [None, input_len]
        # Return: [None, 1]

        # === bias, [None, 1, 1] <= [None, 1]
        bias = self.one_order_embedding_layer(bias_indexes)

        # [None, 1]
        bias = tf.squeeze(bias, axis=-1)

        # === 1d
        # [None, input_len]
        one_order_outputs = tf.multiply(input_values, tf.squeeze(one_order_embedding, -1))

        # [None, 1]
        one_order_outputs = tf.reduce_sum(one_order_outputs, axis=1, keepdims=True)

        # === bias + fm_1d
        # [None, 1]
        fm_1d = one_order_outputs + bias

        return fm_1d

    def _compute_fm_2d(self, two_order_embedding=None,
                       input_values=None):
        # two_order_embedding: [None, input_len, embedding_dim]
        # input_values: [None, input_len]
        # Return: [None, 1]

        # embedding_value = embedding * value, V_{f, i} * x_i
        #   [None, input_len, embedding_dim], use broadcast of tf.
        embedding_value = tf.multiply(two_order_embedding,
                                    tf.expand_dims(input_values, axis=-1))

        # (\sum_i v_{f, i} * x_i) ^ 2,
        # [None, embedding_dim]
        square_of_sum = tf.square(tf.reduce_sum(embedding_value, axis=1))

        # \sum_i ((v_{f, i} * x_i) ^ 2)
        # [None, embedding_dim]
        sum_of_square = tf.reduce_sum(tf.square(embedding_value), axis=1)

        # \sum_f
        # [None, 1]
        fm_2d = 0.5 * tf.reduce_sum(square_of_sum - sum_of_square, axis=-1, keepdims=True)

        return fm_2d

    def _compute_dnn(self, two_order_embedding=None):
        # two_order_embedding: [None, input_len, embedding_dim]
        # Return: [None, 1]

        # [None, input_len, embedding_dim]
        outputs = self.dropout1_layer(two_order_embedding)

        # [None, input_len, embedding_dim]
        outputs = self.dense_layer(outputs)

        # [None, input_len, embedding_dim]
        outputs = self.dropout2_layer(outputs)

        # [None, input_len * embedding_dim]
        outputs = self.flatten_layer(outputs)

        # [None, 1]
        y_dnn = self.dnn_output_layer(outputs)

        return y_dnn

    def call(self, inputs, training=None, mask=None):
        # y = Dense(1)(concat[y_fm_1d, y_fm_2d, y_dnn])

        # input_values: [None, seq_len]
        input_values, input_indexes, bias_indexes = inputs

        # === Embedding
        # [None, input_len, 1] <= [None, input_len]
        one_order_embedding = self.one_order_embedding_layer(input_indexes)

        # [None, input_len, embedding_dim]
        two_order_embedding = self.two_order_embedding_layer(input_indexes)

        # === fm_one_order, [None, 1]
        y_fm_1d = self._compute_fm_1d(bias_indexes=bias_indexes,
                                    one_order_embedding=one_order_embedding,
                                    input_values=input_values)

        # === fm_two_order, [None, 1]
        y_fm_2d = self._compute_fm_2d(two_order_embedding=two_order_embedding,
                                    input_values=input_values)

        # === dnn, [None, 1]
        y_dnn = self._compute_dnn(two_order_embedding=two_order_embedding)

        # [None, 3]
        concat = tf.concat([y_fm_1d, y_fm_2d, y_dnn], axis=-1)

        # [None, 1]
        y_pred = self.output_layer(concat)

        return y_pred


def test_model_once(model=None, input_len=None, input_dim=None):
    batch_size = 2

    input_values = tf.random.uniform((batch_size, input_len))
    input_indexes = tf.random.uniform((batch_size, input_len), minval=0, maxval=input_dim, dtype=tf.int32)
    bias_indexes = tf.constant([[input_dim-1]] * batch_size, dtype=tf.int32)

    # num_user = 6, num_item = 4
    # user: 0~5, item: 6~9
    #input_values = tf.constant(([[1, 1]]), dtype=tf.float32)
    #input_indexes = tf.constant([[4, 7]], dtype=tf.int32)
    #bias_indexes = tf.constant([[10]], dtype=tf.int32)

    inputs = (input_values, input_indexes, bias_indexes)
    outputs = model(inputs)
    return outputs


if __name__ == '__main__':
    num_user = 6
    num_item = 4

    input_len = 2
    input_dim = num_user + num_item
    embedding_dim = 8

    model = DeepFM(input_len=input_len, input_dim=input_dim, embedding_dim=embedding_dim)
    #model.build(input_shape=(None, 448, 448, 3))
    #print(model.summary())
    outputs = test_model_once(model=model, input_len=input_len, input_dim=input_dim)
    print('outputs', outputs.shape)
    print(outputs)