from typing import Any, List

from neuralmonkey.trainers.generic_bandit_trainer import GenericBanditTrainer, \
    BanditObjective, _clip_log_probs
from neuralmonkey.logging import log


import tensorflow as tf

# tests; pylint,mypy


def expected_loss_objective(decoder, k) -> BanditObjective:
    """Get expected loss objective from decoder."""
    return BanditObjective(
        name="{} - expected_loss".format(decoder.name),
        decoder=decoder,
        samples=decoder.sample_ids,
        sample_logprobs=tf.add_n(decoder.sample_logprobs),
        loss=tf.reduce_mean(tf.mul(tf.add_n(decoder.sample_probs),
                                   (1-decoder.rewards)), [0, 1]),
        sample_size=k
    )


def cross_entropy_objective(decoder, k) -> BanditObjective:
    """Get bandit cross-entropy loss objective from decoder."""
    # TODO use k
    return BanditObjective(
        name="{} - cross-entropy".format(decoder.name),
        decoder=decoder,
        samples=decoder.sample_ids,
        sample_logprobs=tf.add_n(decoder.sample_logprobs),
        loss=tf.reduce_mean(tf.mul(tf.add_n(decoder.sample_logprobs),
                                   -decoder.rewards), [0, 1]),
        # TODO clipping?
        sample_size=k
    )


# TODO different kinds of feedback
def pairwise_objective(decoder, k) -> BanditObjective:
    """Get bandit cross-entropy loss objective from decoder."""
    return BanditObjective(
        name="{} - pairwise".format(decoder.name),
        decoder=decoder,
        samples=[decoder.sample_ids, decoder.sample_ids_2],
        sample_logprobs=[tf.add_n(decoder.sample_logprobs),
                         tf.add_n(decoder.sample_logprobs_2)],
        loss=tf.reduce_mean(tf.mul(tf.add_n(decoder.pair_probs),
                                   (1-decoder.rewards)), [0, 1]),  # TODO
        sample_size=k
    )


 # FIXME only 1 decoder/objective so far for ALL bandit objectives
class ExpectedLossTrainer(GenericBanditTrainer):
    def __init__(self, decoders: List[Any], l1_weight=0., l2_weight=0.,
                 learning_rate=1e-4,
                 clip_norm=False, optimizer=None, k=1) -> None:
        objective = expected_loss_objective(decoders[0], k)
        super(ExpectedLossTrainer, self).__init__(
            objective, l1_weight, l2_weight, learning_rate=learning_rate,
            clip_norm=clip_norm,
            optimizer=optimizer)


class CrossEntropyTrainer(GenericBanditTrainer):
    def __init__(self, decoders: List[Any], l1_weight=0., l2_weight=0.,
                 learning_rate=1e-4,
                 clip_norm=False, optimizer=None, k=1) -> None:
        objective = cross_entropy_objective(decoders[0], k)
        super(CrossEntropyTrainer, self).__init__(
            objective, l1_weight, l2_weight, learning_rate=learning_rate,
            clip_norm=clip_norm,
            optimizer=optimizer)


class PairwiseTrainer(GenericBanditTrainer):
    def __init__(self, decoders: List[Any], l1_weight=0., l2_weight=0.,
                 learning_rate=1e-4,
                 clip_norm=False, optimizer=None, k=1) -> None:
        objective = pairwise_objective(decoders[0], k)
        super(PairwiseTrainer, self).__init__(
            objective, l1_weight, l2_weight, learning_rate=learning_rate,
            clip_norm=clip_norm,
            optimizer=optimizer)
