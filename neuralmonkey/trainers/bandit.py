from typing import Any, List

from neuralmonkey.trainers.generic_bandit_trainer import GenericBanditTrainer, \
    BanditObjective

# tests; pylint,mypy

def expected_loss_objective(decoder, k) -> BanditObjective:
    """Get expected loss objective from decoder."""
    return BanditObjective(
        name="{} - expected_loss".format(decoder.name),
        decoder=decoder,
        loss=decoder.sample_batch(k)*decoder.rewards,  # TODO
        gradients=None, # TODO do not auto-compute?
        sample_size=k,
        number_arms=1,
        clip_prob=None
    )

def cross_entropy_objective(decoder, k, clip_prob) -> BanditObjective:
    """Get bandit cross-entropy loss objective from decoder."""
    return BanditObjective(
        name="{} - cross-entropy".format(decoder.name),
        decoder=decoder,
        loss=None,  # TODO
        gradients=None, # TODO
        sample_size=k,
        number_arms=1,
        clip_prob=clip_prob
    )

def pairwise_objective(decoder, k) -> BanditObjective:
    """Get bandit cross-entropy loss objective from decoder."""
    return BanditObjective(
        name="{} - pairwise".format(decoder.name),
        decoder=decoder,
        loss=None,  # TODO
        gradients=None, # TODO
        sample_size=k,
        number_arms=2,
        clip_prob=None
    )


class ExpectedLossTrainer(GenericBanditTrainer):
    def __init__(self, decoders: List[Any], l1_weight=0., l2_weight=0.,
                 learning_rate=1e-4,
                 clip_norm=False, optimizer=None, k=1) -> None:
        objectives = [expected_loss_objective(dec, k) for dec in decoders]
        super(ExpectedLossTrainer, self).__init__(
            objectives, l1_weight, l2_weight, learning_rate=learning_rate,
            clip_norm=clip_norm,
            optimizer=optimizer)


class CrossEntropyTrainer(GenericBanditTrainer):
    def __init__(self, decoders: List[Any], l1_weight=0., l2_weight=0.,
                 learning_rate=1e-4,
                 clip_norm=False, optimizer=None, k=1, clip_prob=0.005) -> None:
        objectives = [cross_entropy_objective(dec, k, clip_prob)
                      for dec in decoders]
        super(CrossEntropyTrainer, self).__init__(
            objectives, l1_weight, l2_weight, learning_rate=learning_rate,
            clip_norm=clip_norm,
            optimizer=optimizer)


class PairwiseTrainer(GenericBanditTrainer):
    def __init__(self, decoders: List[Any], l1_weight=0., l2_weight=0.,
                 learning_rate=1e-4,
                 clip_norm=False, optimizer=None, k=1) -> None:
        objectives = [pairwise_objective(dec, k) for dec in decoders]
        super(PairwiseTrainer, self).__init__(
            objectives, l1_weight, l2_weight, learning_rate=learning_rate,
            clip_norm=clip_norm,
            optimizer=optimizer)
