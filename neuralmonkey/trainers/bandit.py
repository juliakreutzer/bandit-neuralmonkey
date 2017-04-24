from typing import Any, List

import tensorflow as tf

from neuralmonkey.trainers.generic_bandit_trainer import GenericBanditTrainer, \
    BanditObjective
from neuralmonkey.gradient_utils import sum_gradients, scale_gradients, \
    subtract_gradients, divide_gradients, multiply_gradients


# tests; pylint,mypy
def exploit_only_objective(decoder, optimizer) -> \
        BanditObjective:
    """Get exploit only objective from decoder."""
    decoded_logprobs = tf.expand_dims(
        tf.expand_dims(
            tf.reduce_sum(
                tf.pack(decoder.decoded_logprobs), [0]),
            0),
        1)
    decoded = tf.expand_dims(tf.pack(decoder.decoded), 2)
    decoder.neg_sample_ix = tf.constant(-1)  # must be set for fetches
    scalars = tf.stop_gradient(  # don't differentiate this
        # loss from user feedback
        -(decoder.rewards - decoder.baseline)
    )
    scaled_gradients = optimizer.compute_gradients(
        tf.reduce_mean(decoded_logprobs * scalars))

    return BanditObjective(
        name="{} - exploit_only".format(decoder.name),
        decoder=decoder,
        samples=decoded,  # greedy output
        sample_logprobs=decoded_logprobs,
        loss=tf.reduce_mean(tf.mul(decoded_logprobs, -decoder.rewards), [0, 1]),
        gradients=scaled_gradients
    )


def expected_loss_objective(decoder, optimizer, annealing_start=1) \
        -> BanditObjective:
    """Get expected loss objective from decoder."""

    # compute temperature for annealing
    # T(t) = 0.99^t where t = t-t_start if t > t_start
    current_temp = _get_annealing_temp(annealing_start, decoder.step)

    sample_ids, sample_logprobs, _ = _get_samples(decoder, neg=False,
                                                  temp=current_temp)
    decoder.neg_sample_ix = tf.constant(-1)  # not used but needed for outputs

    # compute gradients
    # - (reward-baseline)* gradient(logprobs)
    scalars = tf.stop_gradient(
        -(decoder.rewards - decoder.baseline)
    )  # batch_size x 1
    scaled_gradients = optimizer.compute_gradients(
        tf.reduce_mean(sample_logprobs * scalars))

    return BanditObjective(
        name="{} - expected_loss".format(decoder.name),
        decoder=decoder,
        samples=sample_ids,
        sample_logprobs=sample_logprobs,
        loss=tf.reduce_mean(
            tf.mul(tf.exp(sample_logprobs), -decoder.rewards), [0, 1]),
        gradients=scaled_gradients
    )


def _get_annealing_temp(annealing_start, current_step):
    """ T(t) = 0.99**t if t>start_annealing else 1.0 """
    diff = tf.cast(current_step - annealing_start, tf.float32)
    temp = tf.cond(tf.greater(diff, 0.0),
                   lambda: 0.99**diff,
                   lambda: tf.constant(1.0))
    return temp

def _clip_probs(probs, prob_threshold):
    """ Clip probabilities to some threshold """
    if prob_threshold > 0.00:
        return tf.clip_by_value(probs, clip_value_min=prob_threshold,
                                clip_value_max=1)
    else:
        return probs


def _get_samples(decoder, neg=False, temp=1.0):
    """ Retrieve samples from the model """
    tf.get_variable_scope().reuse_variables()
    sample_mode = decoder.sample_size
    # TODO so far only one sample
    if neg:
        sample_mode *= -1
    _, _, sample_ids, sample_logprob, _, neg_ix = decoder.attention_decoder(
        decoder.embedded_go_symbols,
        attention_on_input=decoder.attention_on_input,
        train_mode=False,
        sample_mode=sample_mode,
        temperature=temp,
        scope="{}/attention_decoder".format(decoder.name))

    # expansion is necessary for generalization to multiple samples
    # time x batch x sample_size
    sample_ids = tf.expand_dims(tf.pack(sample_ids), 2)
    # batch x sample_size
    sample_logprobs = tf.expand_dims(sample_logprob, 1)

    return sample_ids, sample_logprobs, neg_ix


class ExploitOnlyTrainer(GenericBanditTrainer):
    """ Objective without sampling, greedily choosing best output """

    def __init__(self, decoders: List[Any], evaluator, l1_weight=0.,
                 l2_weight=0., clip_norm=False,
                 optimizer=None, binary_feedback=False, store_gradients=False,
                 baseline=False, score_function=False) -> None:
        self.store_gradients = store_gradients
        if score_function:
            raise NotImplementedError("Score function control variate not "
                                      "implemented for ExploitOnlyObjective.")
        objective = exploit_only_objective(decoders[0], optimizer)
        super(ExploitOnlyTrainer, self).__init__(
            objective, evaluator, l1_weight, l2_weight,
            clip_norm=clip_norm,
            optimizer=optimizer, pairwise=False,
            binary_feedback=binary_feedback, store_gradients=store_gradients,
            baseline=baseline)


class ExpectedLossTrainer(GenericBanditTrainer):
    """ EL objective with optional control variates (score function and BL) """

    def __init__(self, decoders: List[Any], evaluator, l1_weight=0.,
                 l2_weight=0., clip_norm=False, annealing_start=1,
                 optimizer=None, binary_feedback=False, store_gradients=False,
                 baseline=False, score_function=False) -> None:
        self.store_gradients = store_gradients
        if score_function:
            raise NotImplementedError(
                "Score function control variate not implemented.")
        else:
            objective = expected_loss_objective(decoders[0], optimizer,
                                                annealing_start=annealing_start)
        super(ExpectedLossTrainer, self).__init__(
            objective, evaluator, l1_weight, l2_weight,
            clip_norm=clip_norm,
            optimizer=optimizer, pairwise=False,
            binary_feedback=binary_feedback,
            store_gradients=store_gradients, baseline=baseline)