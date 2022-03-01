from layers import GraphConvolution, InnerProductDecoder
import tensorflow as tf
from tensorflow.contrib import layers
flags = tf.app.flags
FLAGS = flags.FLAGS


class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg

        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

    def fit(self):
        pass

    def predict(self):
        pass


class MultiEncoder(Model):
    def __init__(self, placeholders, num_features, features_nonzero, **kwargs):
        super(MultiEncoder, self).__init__(**kwargs)

        self.inputs1 = placeholders['features1']
        self.inputs2 = placeholders['features2']
        self.input_dim1 = num_features
        self.input_dim2 = num_features
        self.features_nonzero = features_nonzero
        self.weight = tf.Variable(1.0e-4 * tf.ones(shape=(3025, 3025)), name="weight")
        self.coef = self.weight - tf.matrix_diag(tf.diag_part(self.weight))
        self.adj1 = placeholders['adjs1']
        self.adj2 = placeholders['adjs2']
        self.dropout = placeholders['dropout']
        self.build()

    def _build(self):
        with tf.variable_scope('Encoder1', reuse=None):
            self.hidden1 = GraphConvolution(input_dim=self.input_dim1,
                                            output_dim=FLAGS.hidden1,
                                            adj=self.adj1,
                                            act=tf.nn.relu,
                                            dropout=self.dropout,
                                            logging=self.logging,
                                            name='e_dense_11')(self.inputs1)

            self.noise1 = gaussian_noise_layer(self.hidden1, 0.1)

            self.embeddings1 = GraphConvolution(input_dim=FLAGS.hidden1,
                                                output_dim=FLAGS.hidden2,
                                                adj=self.adj1,
                                                act=lambda x: x,
                                                dropout=self.dropout,
                                                logging=self.logging,
                                                name='e_dense_21')(self.noise1)

            self.z_mean1 = self.embeddings1
            self.ZC1 = tf.matmul(self.coef, self.embeddings1)

            layer_flat1, num_features1 = self.flatten_layer(self.z_mean1)
            layer_full1 = tf.layers.dense(inputs=layer_flat1, units=1024, activation=None,
                                         kernel_initializer=layers.variance_scaling_initializer(dtype=tf.float32))
            self.SZ1 = tf.layers.dense(inputs=layer_full1, units=3, activation=None,
                                     kernel_initializer=layers.variance_scaling_initializer(dtype=tf.float32))

            self.reconstructions1 = InnerProductDecoder(input_dim=FLAGS.hidden2,
                                                        act=lambda x: x,
                                                        logging=self.logging)(self.ZC1)

        with tf.variable_scope('Encoder2', reuse=None):
            self.hidden2 = GraphConvolution(input_dim=self.input_dim2,
                                            output_dim=FLAGS.hidden1,
                                            adj=self.adj2,
                                            act=tf.nn.relu,
                                            dropout=self.dropout,
                                            logging=self.logging,
                                            name='e_dense_12')(self.inputs2)

            self.noise2 = gaussian_noise_layer(self.hidden2, 0.1)

            self.embeddings2 = GraphConvolution(input_dim=FLAGS.hidden1,
                                                output_dim=FLAGS.hidden2,
                                                adj=self.adj2,
                                                act=lambda x: x,
                                                dropout=self.dropout,
                                                logging=self.logging,
                                                name='e_dense_22')(self.noise2)

            self.z_mean2 = self.embeddings2
            self.ZC2 = tf.matmul(self.coef, self.embeddings1)

            layer_flat2, num_features2 = self.flatten_layer2(self.z_mean2)
            layer_full12 = tf.layers.dense(inputs=layer_flat2, units=1024, activation=None,
                                          kernel_initializer=layers.variance_scaling_initializer(dtype=tf.float32))
            self.SZ2 = tf.layers.dense(inputs=layer_full12, units=3, activation=None,
                                       kernel_initializer=layers.variance_scaling_initializer(dtype=tf.float32))

            self.reconstructions2 = InnerProductDecoder(input_dim=FLAGS.hidden2,
                                                        act=lambda x: x,
                                                        logging=self.logging)(self.ZC2)

    def flatten_layer(self, layer):
        layer_shape = layer.get_shape()
        num_features = layer_shape[1:4].num_elements()
        layer_flat = tf.reshape(layer, [-1, num_features])
        return layer_flat, num_features

    def flatten_layer2(self, layer):
        layer_shape = layer.get_shape()
        num_features = layer_shape[1:4].num_elements()
        layer_flat = tf.reshape(layer, [-1, num_features])
        return layer_flat, num_features


def gaussian_noise_layer(input_layer, std):
    noise = tf.random_normal(shape=tf.shape(input_layer), mean=0.0, stddev=std, dtype=tf.float32)
    return input_layer + noise