from typing import cast, Iterable, List, Callable, Optional, Union, Any, \
    Tuple, Dict

import tensorflow as tf
from neuralmonkey.nn.projection import linear
from tensorflow.python.ops import variable_scope as vs



#class NoisyGRUCell(tf.contrib.rnn.RNNCell):
class NoisyGRUCell(tf.nn.rnn_cell.RNNCell):
    """
    Gated Recurrent Unit cell with noise in recursion function.
    It is based on the TensorFlow implementatin of GRU.
    """

    def __init__(self, num_units: int) -> None:
        self._num_units = num_units

    @property
    def output_size(self):
        return self._num_units

    @property
    def state_size(self):
        return self._num_units

    def __call__(self, inputs, state, noise,
                 scope=None) -> Tuple[tf.Tensor, tf.Tensor]:
        """Gated recurrent unit (GRU) with nunits cells."""
        with tf.variable_scope("GRUCell"): #scope or type(self).__name__):  # "GRUCell"
            print(tf.get_variable_scope().name)
            with tf.variable_scope("Gates"):  # Reset gate and update gate.
                input_shape = inputs.get_shape().as_list()[1]
                state_shape = state.get_shape().as_list()[1]
                noise_shape = [state_shape, 2 * self._num_units]
                noise_dist = tf.contrib.distributions.Normal(mu=0., sigma=1.)
                noise_matrix = noise_dist.sample(noise_shape)
                # noise matrix is only half the size of recurrent matrix
                #print("noise matrix {} ".format(noise_matrix))

                print(tf.get_variable_scope())
                linear_transform = linear([inputs, state], 2 * self._num_units, scope=tf.get_variable_scope(),
                                bias=True)

                # first part of noise matrix that transforms input is empty
                noise_empty = tf.zeros([input_shape, 2*self._num_units])
                # second part is sampled from Gaussian
                gradient_val = tf.concat(0, [noise_empty, noise_matrix])
                print([v.name for v in tf.trainable_variables()])
                tf.get_variable_scope().reuse_variables()
                gradient = [(gradient_val, tf.get_variable(
                    "Linear/Matrix", shape=[input_shape+state_shape, 2*self._num_units]))]
                #print("linear transform {}".format(linear_transform))

                if noise:
                    to_add = tf.batch_matmul(state, noise_matrix)
                    linear_transform += to_add

                # We start with bias of 1.0 to not reset and not update.
                r, u = tf.split(1, 2, linear_transform)
                # inputs: batch x embedding_size
                # state: batch x rnn_size
                # r: batch x rnn_size

                r, u = tf.sigmoid(r), tf.sigmoid(u)


            with tf.variable_scope("Candidate"):
                c = tf.tanh(linear([inputs, r * state],
                                   self._num_units, scope=tf.get_variable_scope(), bias=True))
            new_h = u * state + (1 - u) * c  # batch_size x rnn_size
            return new_h, new_h, gradient

            # Wx + Uh_t-1 -> W[x, h_t-1]
            # plus noise: Wx + (U+noise)h_t-1 -> Wx + Uh_t-1 + noise*h_t-1 -> W[x, h_t-1] + noise*h_t-1