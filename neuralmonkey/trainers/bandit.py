from typing import Any, List

from neuralmonkey.trainers.generic_bandit_trainer import GenericBanditTrainer, \
    BanditObjective, _clip_probs, _scale_gradients
from neuralmonkey.logging import log


import tensorflow as tf

# tests; pylint,mypy


def expected_loss_objective(decoder, k) -> BanditObjective:
    """Get expected loss objective from decoder."""
    return BanditObjective(
        name="{} - expected_loss".format(decoder.name),
        decoder=decoder,
        samples=decoder.sample_ids,
        sample_logprobs=decoder.sample_logprobs,
        loss=tf.reduce_mean(tf.mul(decoder.sample_probs, -decoder.rewards),
                             [0, 1]),
        gradients=lambda grad_fun: grad_fun(
            tf.mul(decoder.sample_logprobs, -decoder.rewards)),
        sample_size=k
    )


def cross_entropy_objective(decoder, k, clip_prob, factor) -> BanditObjective:
    """Get bandit cross-entropy loss objective from decoder."""
    # TODO use k
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
                decoder.rewards/(factor*_clip_probs(decoder.sample_probs, clip_prob)))),
        sample_size=k
    )

def pairwise_objective(decoder, k) -> BanditObjective:
    """Get bandit cross-entropy loss objective from decoder."""
    return BanditObjective(
        name="{} - pairwise".format(decoder.name),
        decoder=decoder,
        samples=[decoder.sample_ids, decoder.sample_ids_2],
        sample_logprobs=[decoder.sample_logprobs, decoder.sample_logprobs_2],
        loss=tf.reduce_mean(tf.mul(decoder.pair_probs, -(1-decoder.rewards)),
                             [0, 1]),
        gradients=lambda grad_fun: grad_fun(tf.mul(
            decoder.pair_logprobs, -(1-decoder.rewards))),
        sample_size=k
    )

def pairwise_xent_objective(decoder, k, clip_prob, factor) -> BanditObjective:
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
                decoder.rewards/(factor*_clip_probs(decoder.pair_probs, clip_prob)))),
        sample_size=k
    )


 # FIXME only 1 decoder/objective so far for ALL bandit objectives
class ExpectedLossTrainer(GenericBanditTrainer):
    def __init__(self, decoders: List[Any], evaluator, l1_weight=0., l2_weight=0.,
                 clip_norm=False, optimizer=None, k=1,
                 binary_feedback=False) -> None:
        objective = expected_loss_objective(decoders[0], k)
        super(ExpectedLossTrainer, self).__init__(
            objective, evaluator, l1_weight, l2_weight,
            clip_norm=clip_norm,
            optimizer=optimizer, pairwise=False,
            binary_feedback=binary_feedback)


class CrossEntropyTrainer(GenericBanditTrainer):
    def __init__(self, decoders: List[Any], evaluator, l1_weight=0., l2_weight=0.,
                 clip_norm=False, optimizer=None, k=1,
                 binary_feedback=False, clip_prob=0.0, factor=1e10) -> None:
        objective = cross_entropy_objective(decoders[0], k, clip_prob=clip_prob,
                                            factor=factor)
        super(CrossEntropyTrainer, self).__init__(
            objective, evaluator, l1_weight, l2_weight,
            clip_norm=clip_norm,
            optimizer=optimizer, pairwise=False,
            binary_feedback=binary_feedback)


class PairwiseTrainer(GenericBanditTrainer):
    def __init__(self, decoders: List[Any], evaluator, l1_weight=0., l2_weight=0.,
                 clip_norm=False, optimizer=None, k=1,
                 binary_feedback=False) -> None:
        objective = pairwise_objective(decoders[0], k)
        super(PairwiseTrainer, self).__init__(
            objective, evaluator, l1_weight, l2_weight,
            clip_norm=clip_norm,
            optimizer=optimizer, pairwise=True, binary_feedback=binary_feedback)


class PairwiseXentTrainer(GenericBanditTrainer):
    def __init__(self, decoders: List[Any], evaluator, l1_weight=0., l2_weight=0.,
                 clip_norm=False, optimizer=None, k=1,
                 binary_feedback=False, clip_prob=0.0, factor=1e-10) -> None:
        objective = pairwise_xent_objective(decoders[0], k, clip_prob=clip_prob,
                                            factor=factor)
        super(PairwiseXentTrainer, self).__init__(
            objective, evaluator, l1_weight, l2_weight,
            clip_norm=clip_norm,
            optimizer=optimizer, pairwise=True, binary_feedback=binary_feedback)
