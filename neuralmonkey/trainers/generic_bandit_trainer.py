from typing import Any, Dict, List, NamedTuple, Optional, Tuple
import re

import tensorflow as tf
from neuralmonkey.logging import log


from neuralmonkey.runners.base_runner import (collect_encoders,
                                              BanditExecutable,
                                              BanditExecutionResult,
                                              NextExecute)

# tests: lint, mypy

# pylint: disable=invalid-name
Gradients = List[Tuple[tf.Tensor, tf.Variable]]
BanditObjective = NamedTuple('BanditObjective',
                       [('name', str),
                        ('decoder', Any),
                        ('samples', Any),  # TODO better type
                        ('grad_nondiff', Gradients),
                        ('grad_diff', Gradients),
                        # must have gradients because
                        # loss might not be fully differentiable
                        ('loss_part', Any),
                        ('sample_size', int),
                        ('clip_prob', Optional[float])])

BIAS_REGEX = re.compile(r'[Bb]ias')


# pylint: disable=too-few-public-methods,too-many-locals
class GenericBanditTrainer(object):

    # FIXME
    # only one objective for now

    def __init__(self, objective: BanditObjective,
                 l1_weight=0.0, l2_weight=0.0, learning_rate=1e-4,
                 clip_norm=False, optimizer=None) -> None:

        self.optimizer = optimizer or tf.train.AdamOptimizer(
            learning_rate=learning_rate)
        self.objective = objective

        with tf.variable_scope('regularization'):
            regularizable = [v for v in tf.trainable_variables()
                             if BIAS_REGEX.findall(v.name)]
            l1_value = sum(tf.reduce_sum(abs(v)) for v in regularizable)
            l1_cost = l1_weight * l1_value if l1_weight > 0 else 0.0

            l2_value = sum(tf.reduce_sum(v ** 2) for v in regularizable)
            l2_cost = l2_weight * l2_value if l2_weight > 0 else 0.0

        self.regularizer_cost = l1_cost + l2_cost
        tf.scalar_summary('train_l1', l1_value, collections=["summary_train"])
        tf.scalar_summary('train_l2', l2_value, collections=["summary_train"])

        # sum log probs over time steps (before: list over time steps)
        grad_sum = tf.add_n(self.objective.grad_diff)

        # scale log probs by reward (batch_size x sample_size)
        grad_scaled = tf.mul(self.objective.grad_nondiff, grad_sum)

        # average over batch
        grad_avg = tf.reduce_mean(grad_scaled, [0,1])

        # partial gradients for full sequence
        loss_gradients = self._get_gradients(grad_avg)

        # add gradients for regularization
        regularizer_gradients = _sum_gradients([self._get_gradients(l)
                              for l in [l1_cost, l2_cost] if l != 0.])
        self.gradients = _sum_gradients([regularizer_gradients, loss_gradients])

        # scalar, avg over batch
        self.loss = tf.reduce_mean(tf.mul(tf.add_n(self.objective.loss_part),
                                          self.objective.grad_nondiff), [0, 1])

        self.all_coders = set.union(collect_encoders(self.objective.decoder))

        self.clip_norm = clip_norm

        self.sample_op = self.objective.samples, grad_sum
        self.update_op = self.optimizer.apply_gradients(self.gradients)

        with tf.control_dependencies([self.update_op]):
            self.dummy = tf.constant(0)

        for grad, var in self.gradients:
            if grad is not None:
                tf.histogram_summary('gr_' + var.name,
                                     grad, collections=["summary_gradients"])

        self.histogram_summaries = tf.merge_summary(
            tf.get_collection("summary_gradients"))
        self.scalar_summaries = tf.merge_summary(
            tf.get_collection("summary_train"))

    # only use for computing regularizers
    def _get_gradients(self, tensor: tf.Tensor) -> Gradients:
        gradient_list = self.optimizer.compute_gradients(
            tensor, tf.trainable_variables())
        return gradient_list

    # pylint: disable=unused-argument
    def get_executable(self, update=False, summaries=True) \
            -> BanditExecutable:
        if update:
            log("Update bandit")
            return UpdateBanditExecutable(self.all_coders,
                                          self.objective.decoder.rewards,
                                          self.dummy, self.loss,
                                          #None,
                                          #None)
                                          # TODO fix, uncomment
                                          self.scalar_summaries
                                          if summaries else None,
                                          self.histogram_summaries
                                          if summaries else None)
        else:
            log("Sample bandit")
            return SampleBanditExecutable(self.all_coders,
                                          self.sample_op,
                                          self.regularizer_cost,
                                          None,  # no summaries yet
                                          None)


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
    if prob_threshold > 0.00:
        log_threshold = tf.log(prob_threshold)
        log_max_value = tf.log(1)
        return tf.clip_by_value(log_probs, clip_value_min=log_threshold,
                            clip_value_max=log_max_value)
    else:
        return log_probs


class UpdateBanditExecutable(BanditExecutable):

    def __init__(self, all_coders, reward_placeholder, update_op, loss,
                 scalar_summaries, histogram_summaries):
        self.all_coders = all_coders
        self.reward_placeholder = reward_placeholder
        self.update_op = update_op
        self.loss = loss
        self.scalar_summaries = scalar_summaries
        self.histogram_summaries = histogram_summaries

        self.result = None

    def next_to_execute(self, reward: List[float]) -> NextExecute:
        fetches = {'update_op': self.update_op}
        if self.scalar_summaries is not None:
            fetches['scalar_summaries'] = self.scalar_summaries
            fetches['histogram_summaries'] = self.histogram_summaries
        fetches['loss'] = self.loss
        feedables = self.all_coders
        # extra feed for reward
        return feedables, fetches, {self.reward_placeholder: reward}

    def collect_results(self, results: List[Dict]) -> None:
        if self.scalar_summaries is None:
            scalar_summaries = None
            histogram_summaries = None
        else:
            scalar_summaries = results[0]['scalar_summaries']
            histogram_summaries = results[0]['histogram_summaries']

        self.result = BanditExecutionResult(
            [], loss=results[0]['loss'], scalar_summaries=scalar_summaries,
            histogram_summaries=histogram_summaries,
            image_summaries=None)

    def get_fetches(self):
        fetches = [self.update_op, self.loss]
        if self.scalar_summaries is not None:
            fetches.append(self.scalar_summaries)
        if self.histogram_summaries is not None:
            fetches.append(self.histogram_summaries)
        return fetches

    def get_feeds(self):
        feeds = []
        # reward feed is in additional feed dict
        return feeds

class SampleBanditExecutable(BanditExecutable):

    def __init__(self, all_coders, sample_op, regularization_cost,
                 scalar_summaries, histogram_summaries):
        self.all_coders = all_coders
        self.sample_op = sample_op
        self.scalar_summaries = scalar_summaries
        self.histogram_summaries = histogram_summaries
        self.regularization_cost = regularization_cost

        self.result = None

    def next_to_execute(self, reward=None) -> NextExecute:
        fetches = {'sample_op': self.sample_op}
        if self.scalar_summaries is not None:
            fetches['scalar_summaries'] = self.scalar_summaries
            fetches['histogram_summaries'] = self.histogram_summaries
        fetches['reg_cost'] = self.regularization_cost

        return self.all_coders, fetches, {}

    def collect_results(self, results: List[Dict]) -> None:
        if self.scalar_summaries is None:
            scalar_summaries = None
            histogram_summaries = None
        else:
            scalar_summaries = results[0]['scalar_summaries']
            histogram_summaries = results[0]['histogram_summaries']

        sampled_outputs, sampled_logprobs = results[0]['sample_op']
        reg_cost = results[0]['reg_cost']
        outputs = sampled_outputs, sampled_logprobs, reg_cost  # TODO make summaries for these values
        self.result = BanditExecutionResult(
            [outputs], loss=None,
            scalar_summaries=scalar_summaries,
            histogram_summaries=histogram_summaries,
            image_summaries=None)

    def get_fetches(self):
        fetches = [self.regularization_cost]
        samples, logprobs = self.sample_op
        fetches.extend(samples)
        fetches.append(logprobs)
        if self.scalar_summaries is not None:
            fetches.extend(self.scalar_summaries)
        if self.histogram_summaries is not None:
            fetches.extend(self.histogram_summaries)
        return fetches

    def get_feeds(self):
        feeds = []
        for coder in self.all_coders:
            # need all placeholders of coders
            feeds.extend(coder._get_placeholders())
        return feeds