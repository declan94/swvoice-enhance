import tensorflow as tf
from utils import ioutil
from os.path import join

def _get_weights_for_name(name, n_input, n_hidden):
    all_weights = dict()
    all_weights['encoding_w'] = tf.get_variable(name + "-encoding_w", shape=[n_input, n_hidden], initializer=tf.contrib.layers.xavier_initializer())
    all_weights['encoding_b'] = tf.get_variable(name + "-encoding_b", shape=[n_hidden], initializer=tf.zeros_initializer())
    all_weights['decoding_w'] = tf.get_variable(name + "-decoding_w", shape=[n_hidden, n_input], initializer=tf.contrib.layers.xavier_initializer())
    all_weights['decoding_b'] = tf.get_variable(name + "-decoding_b", shape=[n_input], initializer=tf.zeros_initializer())
    return all_weights

def _get_weights_names(name):
    return [name+"-encoding_w", name+"-encoding_b", name+"-decoding_w", name+"-decoding_b"]

class Autoencoder(object):

    def __init__(self, n_input, n_hidden, transfer_function=tf.nn.softplus, optimizer = tf.train.AdamOptimizer(), name = "autoencoder"):
        super(Autoencoder, self).__init__()

        self.n_input = n_input
        self.n_hidden = n_hidden
        self.transfer = transfer_function
        self.name = name

        network_weights = _get_weights_for_name(name, n_input, n_hidden)
        self.weights = network_weights

        # model
        self.x = tf.placeholder(tf.float32, [None, self.n_input])
        self.y = tf.placeholder(tf.float32, [None, self.n_input])
        self.hidden = self.transfer(tf.add(tf.matmul(self.x, self.weights['encoding_w']), self.weights['encoding_b']))
        self.reconstruction = tf.add(tf.matmul(self.hidden, self.weights['decoding_w']), self.weights['decoding_b'])

        # cost
        self.cost = 0.5 * tf.reduce_mean(tf.pow(tf.subtract(self.reconstruction, self.y), 2.0))
        self.optimizer = optimizer.minimize(self.cost)

        self.saver = tf.train.Saver([self.weights[k] for k in self.weights])

        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

    def __del__( self ):  
         self.sess.close()

    def partial_fit(self, X, Y = None):
        if Y == None:
            Y = X
        cost, opt = self.sess.run((self.cost, self.optimizer), feed_dict={self.x: X, self.y: Y})
        return cost

    def encode(self, X):
        return self.sess.run(self.hidden, feed_dict={self.x: X})

    def generate(self, hidden = None):
        if hidden is None:
            hidden = self.sess.run(tf.random_normal([1, self.n_hidden]))
        return self.sess.run(self.reconstruction, feed_dict={self.hidden: hidden})

    def reconstruct(self, X):
        return self.sess.run(self.reconstruction, feed_dict={self.x: X})

    def get_weights(self):
        weights = {}
        for k in self.weights:
            weights[k] = self.sess.run(self.weights[k])
        return weights

    def save_model(self, path):
        path = join(path, self.name)
        ioutil.ensureDirectory(path)
        self.saver.save(self.sess, path)

    def load_model(self, path):
        path = join(path, self.name)
        self.saver.restore(self.sess, path)

class StackedAE(object):
    """Stacked Autoencoder"""
    
    def __init__(self, n_input, ae_weights = [], transfer_function=tf.nn.softplus, optimizer = tf.train.AdamOptimizer()):
        super(StackedAE, self).__init__()
        self.n_input = n_input
        self.ae_weights = ae_weights
        self.transfer = transfer_function
        self._optimizer = optimizer
        if len(ae_weights) > 0:
            self._buildModel()

    def __del__( self ):  
         self.sess.close()

    def _buildModel(self):
        self.x = tf.placeholder(tf.float32, [None, self.n_input])
        encoding_weights = []
        encoding_layers = []
        last_layer = self.x
        for weights in self.ae_weights:
            w = tf.Variable(weights['encoding_w'], dtype=tf.float32)
            b = tf.Variable(weights['encoding_b'], dtype=tf.float32)
            encoding_weights.append({"w": w, "b": b})
            layer = self.transfer(tf.add(tf.matmul(last_layer, w), b))
            last_layer = layer
            encoding_layers.append(layer)
        self.encoding_weights = encoding_weights
        self.encoding_layers = encoding_layers
        self.encoding_output = last_layer

        self.y = tf.placeholder(tf.float32, [None, self.n_input])
        decoding_weights = []
        decoding_layers = []
        last_layer = self.encoding_output
        for weights in reversed(self.ae_weights):
            w = tf.Variable(weights['decoding_w'], dtype=tf.float32)
            b = tf.Variable(weights['decoding_b'], dtype=tf.float32)
            decoding_weights.append({"w": w, "b": b})
            layer = tf.add(tf.matmul(last_layer, w), b)
            last_layer = layer
            decoding_layers.append(layer)
        self.decoding_weights = decoding_weights
        self.decoding_layers = decoding_layers
        self.decoding_output = last_layer

        # cost
        self.cost = 0.5 * tf.reduce_mean(tf.pow(tf.subtract(self.decoding_output, self.y), 2.0))
        self.optimizer = self._optimizer.minimize(self.cost)

        self.saver = tf.train.Saver()
        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

    def stack_autoencoder(self, ae):
        self.ae_weights.append(ae.get_weights())
        self._buildModel()

    def encode(self, X):
        return self.sess.run(self.encoding_output, feed_dict={self.x: X})

    def reconstruct(self, X):
        return self.sess.run(self.decoding_output, feed_dict={self.x: X})

    def partial_fit(self, X, Y = None):
        if Y == None:
            Y = X
        cost, opt = self.sess.run((self.cost, self.optimizer), feed_dict={self.x: X, self.y: Y})
        return cost

    def save_model(self, path):
        ioutil.ensureDirectory(path)
        self.saver.save(self.sess, path)

    def load_model(self, path):
        self.saver.restore(self.sess, path)

# class AdditiveGaussianNoiseAutoencoder(object):

#     def __init__(self, n_input, n_hidden, transfer_function = tf.nn.softplus, optimizer = tf.train.AdamOptimizer(), scale = 0.1):
#         self.n_input = n_input
#         self.n_hidden = n_hidden
#         self.transfer = transfer_function
#         self.scale = tf.placeholder(tf.float32)
#         self.training_scale = scale
#         network_weights = self._initialize_weights()
#         self.weights = network_weights

#         # model
#         self.x = tf.placeholder(tf.float32, [None, self.n_input])
#         self.hidden = self.transfer(tf.add(tf.matmul(self.x + scale * tf.random_normal((n_input,)),
#                 self.weights['encoding_w']),
#                 self.weights['encoding_b']))
#         self.reconstruction = tf.add(tf.matmul(self.hidden, self.weights['decoding_w']), self.weights['decoding_b'])

#         # cost
#         self.cost = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(self.reconstruction, self.x), 2.0))
#         self.optimizer = optimizer.minimize(self.cost)

#         init = tf.global_variables_initializer()
#         self.sess = tf.Session()
#         self.sess.run(init)

#     def _initialize_weights(self):
#         all_weights = dict()
#         all_weights['encoding_w'] = tf.get_variable("encoding_w", shape=[self.n_input, self.n_hidden],
#             initializer=tf.contrib.layers.xavier_initializer())
#         all_weights['encoding_b'] = tf.Variable(tf.zeros([self.n_hidden], dtype = tf.float32))
#         all_weights['decoding_w'] = tf.Variable(tf.zeros([self.n_hidden, self.n_input], dtype = tf.float32))
#         all_weights['decoding_b'] = tf.Variable(tf.zeros([self.n_input], dtype = tf.float32))
#         return all_weights

#     def partial_fit(self, X):
#         cost, opt = self.sess.run((self.cost, self.optimizer), feed_dict = {self.x: X,
#                                                                             self.scale: self.training_scale
#                                                                             })
#         return cost

#     def calc_total_cost(self, X):
#         return self.sess.run(self.cost, feed_dict = {self.x: X,
#                                                      self.scale: self.training_scale
#                                                      })

#     def transform(self, X):
#         return self.sess.run(self.hidden, feed_dict = {self.x: X,
#                                                        self.scale: self.training_scale
#                                                        })

#     def generate(self, hidden=None):
#         if hidden is None:
#             hidden = self.sess.run(tf.random_normal([1, self.n_hidden]))
#         return self.sess.run(self.reconstruction, feed_dict = {self.hidden: hidden})

#     def reconstruct(self, X):
#         return self.sess.run(self.reconstruction, feed_dict = {self.x: X,
#                                                                self.scale: self.training_scale
#                                                                })

#     def getWeights(self):
#         return self.sess.run(self.weights['encoding_w'])

#     def getBiases(self):
#         return self.sess.run(self.weights['encoding_b'])
