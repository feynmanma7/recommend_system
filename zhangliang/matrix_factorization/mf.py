import tensorflow as tf
tf.random.set_seed(7)
from tensorflow.keras.layers import Embedding


class MF(tf.keras.Model):
    def __init__(self,
                 num_user=100,
                 num_item=50,
                 embedding_dim=16):
        super(MF, self).__init__()

        self.user_embedding_layer = Embedding(input_dim=num_user,
                                              output_dim=embedding_dim)
        self.item_embedding_layer = Embedding(input_dim=num_item,
                                              output_dim=embedding_dim)

    def call(self, inputs=None):
        users, items = inputs
        # users: [None, ]
        # user_embedding: [None, embedding_dim]
        user_embedding = self.user_embedding_layer(users)

        # items: [None, ]
        # item_embedding: [None, embedding_dim]
        item_embedding = self.item_embedding_layer(items)

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
