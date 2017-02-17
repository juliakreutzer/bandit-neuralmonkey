from typing import Any, List

from neuralmonkey.trainers.generic_bandit_trainer import GenericBanditTrainer, \
    BanditObjective, _clip_probs, _scale_gradients
from neuralmonkey.logging import log


import tensorflow as tf

# tests; pylint,mypy


def expected_loss_objective(decoder, initial_temperature) -> BanditObjective:
    """Get expected loss objective from decoder."""
    return BanditObjective(
        name="{} - expected_loss".format(decoder.name),
        decoder=decoder,
        samples=decoder.sample_ids,
        sample_logprobs=decoder.sample_logprobs,
        loss=tf.reduce_mean(tf.mul(decoder.sample_probs, -decoder.rewards),
                             [0, 1]),
        # TODO include entropy in loss
        gradients=lambda grad_fun: _scale_gradients(
            grad_fun(decoder.sample_logprobs),
            -tf.reduce_mean(-decoder.rewards +
             tf.mul(_get_temperature(initial_temperature, decoder.epoch),
                    decoder.sample_logprobs + 1)))
    )


def cross_entropy_objective(decoder, initial_temperature, clip_prob, factor) \
        -> BanditObjective:
    """Get bandit cross-entropy loss objective from decoder."""
    return BanditObjective(
        name="{} - cross-entropy".format(decoder.name),
        decoder=decoder,
        samples=decoder.sample_ids,
        sample_logprobs=decoder.sample_logprobs,
        loss=-tf.reduce_mean(tf.mul(decoder.sample_logprobs, decoder.rewards),
                             [0, 1]),
        gradients=lambda grad_fun: _scale_gradients(
            grad_fun(decoder.sample_logprobs),
            -tf.reduce_mean(
                (decoder.rewards - tf.mul(_get_temperature(initial_temperature, decoder.epoch),
                                          decoder.sample_logprobs + 1))/
                (factor*_clip_probs(decoder.sample_probs, clip_prob))))
    )


def pairwise_objective(decoder, initial_temperature) -> BanditObjective:
    """Get bandit cross-entropy loss objective from decoder."""
    return BanditObjective(
        name="{} - pairwise".format(decoder.name),
        decoder=decoder,
        samples=[decoder.sample_ids, decoder.sample_ids_2],
        sample_logprobs=[decoder.sample_logprobs, decoder.sample_logprobs_2],
        loss=tf.reduce_mean(tf.mul(decoder.pair_probs, -(1-decoder.rewards)),
                             [0, 1]),
        gradients=lambda grad_fun: _scale_gradients(
            grad_fun(decoder.pair_logprobs),
            tf.reduce_mean(-(1-decoder.rewards) +
            tf.mul(_get_temperature(initial_temperature, decoder.epoch),
                                            decoder.sample_logprobs + 1)))
    )

def pairwise_xent_objective(decoder, initial_temperature, clip_prob, factor) \
        -> BanditObjective:
    """Get bandit cross-entropy loss objective from decoder."""
    return BanditObjective(
        name="{} - pairwise_xent".format(decoder.name),
        decoder=decoder,
        samples=[decoder.sample_ids, decoder.sample_ids_2],
        sample_logprobs=[decoder.sample_logprobs,
                         decoder.sample_logprobs_2],
        loss=-tf.reduce_mean(tf.mul(decoder.pair_logprobs,
                                    decoder.rewards), [0, 1]),
        gradients=lambda grad_fun: _scale_gradients(
            grad_fun(decoder.pair_logprobs),
            -tf.reduce_mean(
                (decoder.rewards - tf.mul(_get_temperature(initial_temperature, decoder.epoch),
                                          decoder.sample_logprobs + 1))/
                 (factor*_clip_probs(decoder.pair_probs, clip_prob))))
    )


def _get_temperature(initial_temperature, current_epoch):
    """
    Annealing temperature with decay function as in ACL paper:
    T = T0 / ((epoch + 1)^1/3)
    :param initial_temperature:
    :param current_epoch:
    :return:
    """
    return initial_temperature/((tf.cast(current_epoch, tf.float32)+1)**1/3.)


class ExpectedLossTrainer(GenericBanditTrainer):
    def __init__(self, decoders: List[Any], evaluator, l1_weight=0.,
                 l2_weight=0., initial_temperature=0., clip_norm=False,
                 optimizer=None, binary_feedback=False) -> None:
        initial_temperature = initial_temperature
        objective = expected_loss_objective(decoders[0],
                                            initial_temperature=initial_temperature)
        super(ExpectedLossTrainer, self).__init__(
            objective, evaluator, l1_weight, l2_weight,
            clip_norm=clip_norm,
            optimizer=optimizer, pairwise=False,
            binary_feedback=binary_feedback)


class CrossEntropyTrainer(GenericBanditTrainer):
    def __init__(self, decoders: List[Any], evaluator, l1_weight=0.,
                 l2_weight=0., initial_temperature=0., clip_norm=False,
                 optimizer=None, binary_feedback=False,
                 clip_prob=0.0, factor=1e10) -> None:
        objective = cross_entropy_objective(decoders[0],
                                            initial_temperature=initial_temperature,
                                            clip_prob=clip_prob,
                                            factor=factor)
        super(CrossEntropyTrainer, self).__init__(
            objective, evaluator, l1_weight, l2_weight,
            clip_norm=clip_norm,
            optimizer=optimizer, pairwise=False,
            binary_feedback=binary_feedback)


class PairwiseTrainer(GenericBanditTrainer):
    def __init__(self, decoders: List[Any], evaluator, l1_weight=0.,
                 l2_weight=0., initial_temperature=0., clip_norm=False,
                 optimizer=None, binary_feedback=False) -> None:
        objective = pairwise_objective(decoders[0],
                                       initial_temperature=initial_temperature)
        super(PairwiseTrainer, self).__init__(
            objective, evaluator, l1_weight, l2_weight, clip_norm=clip_norm,
            optimizer=optimizer, pairwise=True, binary_feedback=binary_feedback)


class PairwiseXentTrainer(GenericBanditTrainer):
    def __init__(self, decoders: List[Any], evaluator, l1_weight=0.,
                 l2_weight=0., initial_temperature=0., clip_norm=False,
                 optimizer=None, binary_feedback=False,
                 clip_prob=0., factor=1.) -> None:
        objective = pairwise_xent_objective(decoders[0],
                                            initial_temperature=initial_temperature,
                                            clip_prob=clip_prob,
                                            factor=factor)
        super(PairwiseXentTrainer, self).__init__(
            objective, evaluator, l1_weight, l2_weight,
            clip_norm=clip_norm,
            optimizer=optimizer, pairwise=True, binary_feedback=binary_feedback)
