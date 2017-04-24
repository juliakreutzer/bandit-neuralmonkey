from typing import Dict, List, Tuple

import tensorflow as tf

Gradients = List[Tuple[tf.Tensor, tf.Variable]]


def get_gradients(optimizer, tensor: tf.Tensor) -> Gradients:
    gradient_list = optimizer.compute_gradients(
        tensor, tf.trainable_variables(),
        aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N)
    return gradient_list


def init_grad():
    gradient = []
    for var in tf.trainable_variables():
        gradient.append((tf.zeros_like(var, dtype=tf.float32), var))
    return gradient


def add_to_gradients(gradients: Gradients, scalar: tf.float32) -> Gradients:
    """ Add a scalar to every component of the given gradient """
    sum_dict = {}
    for tensor, var in gradients:
        sum_dict[var] = tensor + scalar
    return [(tensor, var) for var, tensor in sum_dict.items()]


def sum_gradients(gradients_list: List[Gradients]) -> Gradients:
    """ Sum a list of gradients component-wise """
    summed_dict = {}  # type: Dict[tf.Variable, tf.Tensor]
    for gradients in gradients_list:
        for tensor, var in gradients:
            if tensor is not None:
                if not var in summed_dict:
                    summed_dict[var] = tensor
                else:
                    summed_dict[var] += tensor
    return [(tensor, var) for var, tensor in summed_dict.items()]


def multiply_gradients(gradients_list: List[Gradients]) -> Gradients:
    """ Multiply a list of gradients component-wise """
    product_dict = {}  # type: Dict[tf.Variable, tf.Tensor]
    for gradients in gradients_list:
        for tensor, var in gradients:
            if tensor is not None:
                if not var in product_dict:
                    product_dict[var] = tensor
                else:
                    product_dict[var] *= tensor
    return [(tensor, var) for var, tensor in product_dict.items()]


def subtract_gradients(x: Gradients, y: Gradients) -> Gradients:
    """ Subtract the components of y from x """
    subtract_dict = {}  # type: Dict[tf.Variable, tf.Tensor]
    for tensor, var in x:
        subtract_dict[var] = tensor
    for tensor, var in y:
        subtract_dict[var] -= tensor
    return [(tensor, var) for var, tensor in subtract_dict.items()]


def scale_gradients(gradients: [Gradients], scalar) -> Gradients:
    """ Scale the components of a gradient """
    scaled_dict = {}  # type: Dict[tf.Variable, tf.Tensor]
    for tensor, var in gradients:
        if tensor is not None:
            scaled_dict[var] = tensor*scalar
    return [(tensor, var) for var, tensor in scaled_dict.items()]


def divide_gradients(x: Gradients, y: Gradients) -> Gradients:
    """ Divide the components of x by the components of y """
    divide_dict = {}  # type: Dict[tf.Variable, tf.Tensor]
    for tensor, var in x:
        divide_dict[var] = tensor
    for tensor, var in y:
        divide_dict[var] -= tensor
    return [(tensor, var) for var, tensor in divide_dict.items()]