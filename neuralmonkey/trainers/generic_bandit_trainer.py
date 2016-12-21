from typing import Any, Dict, List, NamedTuple, Optional, Tuple
import re

import tensorflow as tf

from neuralmonkey.runners.base_runner import (collect_encoders, Executable,
                                              ExecutionResult, NextExecute)

# tests: lint, mypy

# pylint: disable=invalid-name
Gradients = List[Tuple[tf.Tensor, tf.Variable]]
BanditObjective = NamedTuple('BanditObjective',
                       [('name', str),
                        ('decoder', Any),
                        ('loss', tf.Tensor),
                        ('gradients', Gradients),
                        # must have gradients because
                        # loss might not be fully differentiable
                        ('sample_size', int),
                        ('clip_prob', Optional[float])])

BIAS_REGEX = re.compile(r'[Bb]ias')


# pylint: disable=too-few-public-methods,too-many-locals
class GenericBanditTrainer(object):

    # TODO
    # get sample and its probability
    # compute gradient of sample w.r.t to weights
    # get loss from outside graph
    # compute stochastic gradient
    # update model

    def __init__(self, objectives: List[BanditObjective],
                 l1_weight=0.0, l2_weight=0.0, learning_rate=1e-4,
                 clip_norm=False, optimizer=None) -> None:

        # reward needs to be computed outside the TF
        self.rewards = tf.placeholder(tf.float32, [None])

        for obj in objectives:
            obj.decoder.rewards = self.rewards  # TODO does that make sense?

        self.optimizer = optimizer(learning_rate=learning_rate) or \
                         tf.train.AdamOptimizer(learning_rate=learning_rate)

        with tf.variable_scope('regularization'):
            regularizable = [v for v in tf.trainable_variables()
                             if BIAS_REGEX.findall(v.name)]
            l1_value = sum(tf.reduce_sum(abs(v)) for v in regularizable)
            l1_cost = l1_weight * l1_value if l1_weight > 0 else 0.0

            l2_value = sum(tf.reduce_sum(v ** 2) for v in regularizable)
            l2_cost = l2_weight * l2_value if l2_weight > 0 else 0.0

        self.losses = [o.loss for o in objectives] + [l1_value, l2_value]
        tf.scalar_summary('train_l1', l1_value, collections=["summary_train"])
        tf.scalar_summary('train_l2', l2_value, collections=["summary_train"])

        partial_gradients = []  # type: List[Gradients]
        for objective in objectives:
            partial_gradients.append(objective.gradients)
        partial_gradients += [self._get_gradients(l)
                              for l in [l1_cost, l2_cost] if l != 0.]

        gradients = _sum_gradients(partial_gradients)

        if clip_norm:
            assert clip_norm > 0.0
            gradients = [(tf.clip_by_norm(grad, clip_norm), var)
                         for grad, var in gradients]

        self.all_coders = set.union(*(collect_encoders(obj.decoder)
                                      for obj in objectives))
        self.train_op = self.optimizer.apply_gradients(gradients)

        for grad, var in gradients:
            if grad is not None:
                tf.histogram_summary('gr_' + var.name,
                                     grad, collections=["summary_gradients"])

        self.histogram_summaries = tf.merge_summary(
            tf.get_collection("summary_gradients"))
        self.scalar_summaries = tf.merge_summary(
            tf.get_collection("summary_train"))

    # only use for computing regularizers
    def _get_gradients(self, tensor: tf.Tensor) -> Gradients:
        gradient_list = self.optimizer.compute_gradients(tensor)
        return gradient_list

    # pylint: disable=unused-argument
    def get_executable(self, train=False, summaries=True) -> Executable:
        return TrainBanditExecutable(self.all_coders,
                               self.train_op,
                               self.losses,
                               self.scalar_summaries if summaries else None,
                               self.histogram_summaries if summaries else None)


def _sum_gradients(gradients_list: List[Gradients]) -> Gradients:
    summed_dict = {}  # type: Dict[tf.Variable, tf.Tensor]
    for gradients in gradients_list:
        for tensor, var in gradients:
            if tensor is not None:
                if not var in summed_dict:
                    summed_dict[var] = tensor
                else:
                    summed_dict[var] += tensor
    return [(tensor, var) for var, tensor in summed_dict.items()]


def _clip_log_probs(log_probs, prob_threshold):
    """ Clip log probabilities to some threshold """
    # threshold is prob, input are log probs
    log_threshold = tf.log(prob_threshold)
    log_max_value = tf.log(1)
    return tf.clip_by_value(log_probs, clip_value_min=log_threshold,
                            clip_value_max=log_max_value)


class TrainBanditExecutable(Executable):

    def __init__(self, all_coders, train_op, losses, scalar_summaries,
                 histogram_summaries):
        self.all_coders = all_coders
        self.train_op = train_op
        self.losses = losses
        self.scalar_summaries = scalar_summaries
        self.histogram_summaries = histogram_summaries

        self.result = None

    def next_to_execute(self) -> NextExecute:
        fetches = {'train_op': self.train_op}
        if self.scalar_summaries is not None:
            fetches['scalar_summaries'] = self.scalar_summaries
            fetches['histogram_summaries'] = self.histogram_summaries
        fetches['losses'] = self.losses

        return self.all_coders, fetches, {}

    def collect_results(self, results: List[Dict]) -> None:
        if self.scalar_summaries is None:
            scalar_summaries = None
            histogram_summaries = None
        else:
            # TODO collect summaries from different sessions
            scalar_summaries = results[0]['scalar_summaries']
            histogram_summaries = results[0]['histogram_summaries']

        losses_sum = [0. for _ in self.losses]
        for session_result in results:
            for i in range(len(self.losses)):
                # from the end, losses are last ones
                losses_sum[i] += session_result['losses'][i]
        avg_losses = [s / len(results) for s in losses_sum]

        self.result = ExecutionResult(
            [], losses=avg_losses,
            scalar_summaries=scalar_summaries,
            histogram_summaries=histogram_summaries,
            image_summaries=None)
