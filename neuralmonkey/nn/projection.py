"""
This module implements various types of projections.
"""
# tests: lint, mypy
import tensorflow as tf

from neuralmonkey.logging import log


def linear(inputs, output_size, scope="LinearProjection", bias=True, neg=False):
    """Simple linear projection

    y = Wx + b

    Also provides result of projection with negative results

    Arguments:
        inputs: A tensor. It should be a 2D tensor with
                equal length in the first dimension (batch size)
        output_size: The size of dimension 1 of the output tensor.
        scope: The name of the scope used for the variables.
        bias: Whether to add bias or not.
        neg: Whether to use negated weights or not.

    Returns:
        A tensor of shape batch x size
    """
    shapes = inputs.get_shape().as_list()
    with tf.variable_scope(scope):
        # Now the computation.
        with tf.variable_scope(scope or "Linear"):
            matrix = tf.get_variable(
                "Matrix", [shapes[1], output_size], dtype=tf.float32)
            matrix_neg = tf.mul(-1., matrix)
            res = tf.matmul(inputs, matrix)
            res_neg = tf.matmul(inputs, matrix_neg)
            if not bias:
                if not neg:
                    return res
                else:
                    return res_neg
            bias_term = tf.get_variable(
                "Bias", [output_size],
                dtype=tf.float32,
                initializer=tf.constant_initializer(
                    0.0, dtype=tf.float32))
        if not neg:
            return res + bias_term
        else:
            return res_neg + bias_term
        #return tf.nn.seq2seq.linear(inputs, size, True)


def nonlinear(inputs, size, activation=tf.tanh, scope="NonlinearProjection",
              bias=True, neg=False):
    """Linear projection with non-linear activation function

    y = activation(Wx + b)

    Arguments:
        inputs: A tensor or list of tensors. It should be 2D tensors
                with equal length in the first dimension (batch size)
        size: The size of the second dimension (index 1) of the output tensor
        scope: The name of the scope used for the variables

    Returns:
        A tensor of shape batch x size
    """
    with tf.variable_scope(scope) as varscope:
        return activation(linear(inputs, size, scope=varscope,
                                 bias=bias, neg=neg))


def maxout(inputs, size, scope="MaxoutProjection"):
    """Implementation of Maxout layer (Goodfellow et al., 2013)
    http://arxiv.org/pdf/1302.4389.pdf

    z = Wx + b
    y_i = max(z_{2i-1}, z_{2i})

    Arguments:
        inputs: A tensor or list of tensors. It should be 2D tensors with
                equal length in the first dimension (batch size)
        size: The size of dimension 1 of the output tensor.
        scope: The name of the scope used for the variables

    Returns:
        A tensor of shape batch x size
    """
    with tf.variable_scope(scope):
        projected = linear(inputs, size * 2, scope=scope)
        maxout_input = tf.reshape(projected, [-1, 1, 2, size])
        maxpooled = tf.nn.max_pool(
            maxout_input, [1, 1, 2, 1], [1, 1, 2, 1], "SAME")

        reshaped = tf.reshape(maxpooled, [-1, size])
        return reshaped
