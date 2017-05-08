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

    def __call__(self, inputs, state, noise_recurrent=None,
                 scope=None) -> Tuple[tf.Tensor, tf.Tensor]:
        """Gated recurrent unit (GRU) with nunits cells."""
        with tf.variable_scope("GRUCell"): #scope or type(self).__name__):  # "GRUCell"
            with tf.variable_scope("Gates"):  # Reset gate and update gate.
                input_shape = inputs.get_shape().as_list()[1]
                state_shape = self._num_units
                linear_transform = linear([inputs, state], 2 * self._num_units,
                                          scope=tf.get_variable_scope(),
                                          bias=True)
                gradient = None
                if noise_recurrent is not None:
                    # first part of noise matrix that transforms input is empty
                    #noise_empty = tf.zeros([input_shape, 2*self._num_units])
                    # second part is sampled from Gaussian
                    tf.get_variable_scope().reuse_variables()

                    mean, var = tf.nn.moments(tf.get_variable("Linear/Matrix",
                                                 shape=[input_shape+state_shape,
                                                        2*self._num_units]),
                                              axes=[0, 1])
                    # batch-normalize noise according to mean and var of matrix
                    noise_recurrent = tf.nn.batch_normalization(
                        noise_recurrent, mean=mean, variance=var, offset=None,
                        scale=None, variance_epsilon=1.0e-5)

                    #gradient_val = tf.concat(0, [noise_empty, noise_recurrent])
                    gradient_val = noise_recurrent
                    gradient = [(gradient_val,
                                 tf.get_variable("Linear/Matrix",
                                                 shape=[input_shape+state_shape,
                                                        2*self._num_units]))]

                    # noise to both recurrent matrices
                    input_and_state = tf.concat(1, [inputs, state])
                    to_add = tf.batch_matmul(input_and_state, noise_recurrent)
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