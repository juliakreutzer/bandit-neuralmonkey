from typing import Any, Dict, List, NamedTuple, Optional, Tuple
import re

import tensorflow as tf
from neuralmonkey.logging import log

import numpy as np


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
                        ('sample_logprobs', Any),
                        ('loss', Any),
                        ('gradients', Any)])

BIAS_REGEX = re.compile(r'[Bb]ias')


# pylint: disable=too-few-public-methods,too-many-locals
class GenericBanditTrainer(object):

    # only one objective for now

    def __init__(self, objective: BanditObjective, evaluator,
                 l1_weight=0.0, l2_weight=0.0,
                 clip_norm=False, optimizer=None, pairwise=False,
                 binary_feedback=False, number_of_samples=1, store_gradients=False,
                 baseline=False)\
            -> None:

        with tf.name_scope("trainer"):

            self.optimizer = optimizer or tf.train.AdamOptimizer(1e-4)
            self.objective = objective

            self.pairwise = pairwise
            self.binary_feedback = binary_feedback

            self.evaluator = evaluator
            self.baseline = baseline

            print([v.name for v in tf.trainable_variables()])

            with tf.variable_scope('regularization'):
                regularizable = [v for v in tf.trainable_variables()
                                 if BIAS_REGEX.findall(v.name)]
                l1_value = sum(tf.reduce_sum(abs(v)) for v in regularizable)
                l1_cost = l1_weight * l1_value \
                    if l1_weight > 0 else tf.constant(0.0)

                l2_value = sum(tf.reduce_sum(v ** 2) for v in regularizable)
                l2_cost = l2_weight * l2_value \
                    if l2_weight > 0 else tf.constant(0.0)

            self.regularizer_cost = l1_cost + l2_cost
            tf.scalar_summary('train_l1', l1_value,
                              collections=["summary_train"])
            tf.scalar_summary('train_l2', l2_value,
                              collections=["summary_train"])

            # loss is scalar, avg over batch
            self.loss = self.objective.loss + self.regularizer_cost

            # compute and apply gradients
            self.gradients = self.objective.gradients(self._get_gradients)
            self.reg_gradients = self._get_gradients(self.regularizer_cost)
            self.all_gradients = \
                _sum_gradients([self.gradients, self.reg_gradients])

            if clip_norm:
                assert clip_norm > 0.0
                self.all_gradients = [(tf.clip_by_norm(grad, clip_norm), var)
                             for grad, var in self.all_gradients
                             if grad is not None]

            self.all_coders = set.union(
                collect_encoders(self.objective.decoder))

            self.clip_norm = clip_norm

            self.sample_op = self.objective.samples, \
                             self.objective.sample_logprobs, \
                             self.objective.decoder.neg_sample_ix

            self.greedy_op = self.objective.decoder.decoded

            self.update_op = self.optimizer.apply_gradients(self.all_gradients)

            # hack: partial run requires Tensor as output of operation
            with tf.control_dependencies([self.update_op]):
                self.dummy = tf.constant(0)

            for grad, var in self.all_gradients:
                if grad is not None:
                    tf.histogram_summary('gr_' + var.name, grad,
                                         collections=["summary_gradients"])



            if store_gradients:
                # prepare stochastic gradients to be fetched them from the graph
                # sort gradients by variable name
                sorted_gradients = sorted(self.all_gradients,
                                          key=lambda tup: tup[1].name)

                # flatten the gradients to one big vector to track them
                flattened_gradients = []
                flattened_updates = []

                for grad, var in sorted_gradients:

                    if isinstance(optimizer, tf.train.AdamOptimizer):
                        lr_t = optimizer._lr
                        m_t = optimizer._beta1 * optimizer.get_slot(var, "m") + \
                              (1-optimizer._beta1) * grad
                        v_t = optimizer._beta2 * optimizer.get_slot(var, "v") + \
                              (1-optimizer._beta2) * (grad*grad)
                        var_update = - lr_t * m_t / (tf.sqrt(v_t) + optimizer._epsilon)

                    elif isinstance(optimizer, tf.train.GradientDescentOptimizer):
                        lr_t = optimizer._learning_rate
                        var_update = - lr_t * grad

                    elif isinstance(optimizer, tf.train.MomentumOptimizer):
                        lr_t = optimizer._learning_rate
                        v_t = optimizer.get_slot(var, "momentum")
                        accum = v_t * optimizer._momentum + grad
                        var_update = - lr_t * accum

                    # TODO implement this for adadelta etc.
                    else:
                        raise NotImplementedError(
                            "Gradient tracking notimplemented for {}".format(
                                optimizer))

                    flattened_gradients.append(tf.reshape(grad,[-1]))
                    flattened_updates.append(tf.reshape(var_update, [-1]))

                self.concatenated_gradients = tf.concat(0, flattened_gradients)
                self.concatenated_updates = tf.concat(0, flattened_updates)

            self.histogram_summaries = tf.merge_summary(
                tf.get_collection("summary_gradients"))
            self.scalar_summaries = tf.merge_summary(
                tf.get_collection("summary_train"))

    def _get_gradients(self, tensor: tf.Tensor) -> Gradients:
        gradient_list = self.optimizer.compute_gradients(
            tensor, tf.trainable_variables(), aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N)
        return gradient_list

    # pylint: disable=unused-argument
    def get_executable(self, update=False, summaries=True, store_gradients=False) \
            -> BanditExecutable:
        outputs = [self.concatenated_gradients, self.concatenated_updates] if store_gradients else []
        if update:
            return UpdateBanditExecutable(self.all_coders,
                                          [self.objective.decoder.rewards, self.objective.decoder.baseline],
                                          self.objective.decoder.epoch,
                                          self.dummy, self.loss,
                                          outputs,
                                          self.scalar_summaries
                                          if summaries else None,
                                          self.histogram_summaries
                                          if summaries else None,
                                          store_gradients=store_gradients)
        else:
            return SampleBanditExecutable(self.all_coders,
                                          self.sample_op,
                                          self.greedy_op,
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


def _clip_probs(probs, prob_threshold):
    """ Clip probabilities to some threshold """
    if prob_threshold > 0.00:
        log("Clipping probs <= {}".format(prob_threshold))
        return tf.clip_by_value(probs, clip_value_min=prob_threshold,
                            clip_value_max=1)
    else:
        return probs


class UpdateBanditExecutable(BanditExecutable):

    def __init__(self, all_coders, reward_placeholder, epoch_placeholder,
                 update_op, loss, gradient, scalar_summaries, histogram_summaries, store_gradients):
        self.all_coders = all_coders
        self.reward_placeholder, self.baseline_placeholder = reward_placeholder
        self.epoch_placeholder = epoch_placeholder
        self.update_op = update_op
        self.loss = loss
        self.scalar_summaries = scalar_summaries
        self.histogram_summaries = histogram_summaries
        self.gradient = gradient
        self.store_gradients = store_gradients

        self.result = None

    def next_to_execute(self, reward: List[float], baseline: float, epoch: int) -> NextExecute:
        fetches = {'update_op': self.update_op}
        if self.scalar_summaries is not None:
            fetches['scalar_summaries'] = self.scalar_summaries
            fetches['histogram_summaries'] = self.histogram_summaries
        fetches['loss'] = self.loss
        if self.store_gradients:
            fetches['gradient'] = self.gradient
        feedables = self.all_coders
        # extra feed for reward
        return feedables, fetches, {self.reward_placeholder: reward,
                                    self.epoch_placeholder: epoch,
                                    self.baseline_placeholder: baseline}

    def collect_results(self, results: List[Dict]) -> None:
        if self.scalar_summaries is None:
            scalar_summaries = None
            histogram_summaries = None
        else:
            scalar_summaries = results[0]['scalar_summaries']
            histogram_summaries = results[0]['histogram_summaries']

        gradient_result = results[0]['gradient'] if self.store_gradients else []

        self.result = BanditExecutionResult(
            [gradient_result], loss=results[0]['loss'],
            scalar_summaries=scalar_summaries,
            histogram_summaries=histogram_summaries,
            image_summaries=None)

    def get_fetches(self):
        fetches = [self.update_op, self.loss]
        if self.store_gradients:
            fetches.append(self.gradient)
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

    def __init__(self, all_coders, sample_op, greedy_op, regularization_cost,
                 scalar_summaries, histogram_summaries):
        self.all_coders = all_coders
        self.sample_op = sample_op
        self.greedy_op = greedy_op
        self.scalar_summaries = scalar_summaries
        self.histogram_summaries = histogram_summaries
        self.regularization_cost = regularization_cost
        self.result = None

    def next_to_execute(self, reward=None, baseline=None, epoch=None) -> NextExecute:
        fetches = {'sample_op': self.sample_op}
        fetches["greedy_op"] = self.greedy_op
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

        sampled_outputs, sampled_logprobs, neg_sample_ix = results[0]['sample_op']
        greedy_outputs = results[0]['greedy_op']
        reg_cost = results[0]['reg_cost']
        outputs = sampled_outputs, greedy_outputs, sampled_logprobs, reg_cost, neg_sample_ix
        # TODO make summaries for these values
        self.result = BanditExecutionResult(
            [outputs], loss=None,
            scalar_summaries=scalar_summaries,
            histogram_summaries=histogram_summaries,
            image_summaries=None)

    def get_fetches(self):
        fetches = [self.regularization_cost]
        samples, logprobs, ix = self.sample_op
        greedy = self.greedy_op
        fetches.append(samples)
        fetches.append(logprobs)
        fetches.append(greedy)
        fetches.append(ix)
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
