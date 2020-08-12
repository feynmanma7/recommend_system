import tensorflow as tf
tf.random.set_seed(7)
from tensorflow.keras.layers import Embedding, Dropout, BatchNormalization


class MF(tf.keras.Model):
    def __init__(self,
                 num_user=100,
                 num_item=50,
                 embedding_dim=16,
                 dropout_ratio=0.5):
        super(MF, self).__init__()

        self.user_embedding_layer = Embedding(input_dim=num_user,
                                              output_dim=embedding_dim)
        self.item_embedding_layer = Embedding(input_dim=num_item,
                                              output_dim=embedding_dim)
        #self.dropout_layer_1 = Dropout(rate=dropout_ratio)
        #self.dropout_layer_2 = Dropout(rate=dropout_ratio)

        self.bn_layer_1 = BatchNormalization()
        self.bn_layer_2 = BatchNormalization()


    def call(self, inputs=None):
        users, items = inputs
        # users: [None, ]
        # user_embedding: [None, embedding_dim]
        user_embedding = self.user_embedding_layer(users)

        #user_embedding = self.dropout_layer_1(user_embedding)
        user_embedding = self.bn_layer_1(user_embedding)

        # items: [None, ]
        # item_embedding: [None, embedding_dim]
        item_embedding = self.item_embedding_layer(items)

        #item_embedding = self.dropout_layer_2(item_embedding)
        item_embedding = self.bn_layer_2(item_embedding)

        # [None, embedding]
        sim = tf.multiply(user_embedding, item_embedding)

        # [None, ]
        #sim = tf.reduce_mean(sim, axis=[1]) # What's the difference ?!!
        sim = tf.reduce_mean(sim, axis=-1)

        #print('mf, sim', sim.shape)
        return sim


def test_model_once(model=None, num_user=None, num_item=None):
    batch_size = 2
    users = tf.random.uniform((batch_size, ), minval=0, maxval=num_user, dtype=tf.int32)
    items = tf.random.uniform((batch_size, ), minval=0, maxval=num_item, dtype=tf.int32)

    sim = model(inputs=(users, items))
    return sim


if __name__ == '__main__':
    num_user = 100
    num_item = 50
    embedding_dim = 16

    model = MF(num_user=num_user, num_item=num_item, embedding_dim=embedding_dim)

    outputs = test_model_once(model=model, num_user=num_user, num_item=num_item)
    print(outputs.shape)
