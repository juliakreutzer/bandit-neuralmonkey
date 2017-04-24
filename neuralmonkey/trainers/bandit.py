from typing import Any, List

import tensorflow as tf

from neuralmonkey.trainers.generic_bandit_trainer import GenericBanditTrainer, \
    BanditObjective
from neuralmonkey.gradient_utils import sum_gradients, scale_gradients, \
    subtract_gradients, divide_gradients, multiply_gradients


# tests; pylint,mypy


def exploit_only_objective(decoder, optimizer, initial_temperature) -> \
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
        # entropy regularizer T*(log p +1)
        + _get_temperature(initial_temperature,
                           decoder.epoch)
        * (decoded_logprobs + 1)
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


def expected_loss_objective(decoder, optimizer, initial_temperature) \
        -> BanditObjective:
    """Get expected loss objective from decoder."""
    sample_ids, sample_logprobs, _ = _get_samples(decoder, neg=False)
    decoder.neg_sample_ix = tf.constant(-1)  # not used but needed for outputs

    # compute gradients
    # - (reward-baseline)* gradient(logprobs)
    scalars = tf.stop_gradient(
        -(decoder.rewards - decoder.baseline) +
        _get_temperature(initial_temperature,
                         decoder.epoch)
        * (sample_logprobs + 1)
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


def expected_loss_objective_with_scorefun(decoder, optimizer) -> \
        BanditObjective:
    """Get expected loss objective with score function control variate."""
    sample_ids, sample_logprobs, _ = _get_samples(decoder, neg=False)
    decoder.neg_sample_ix = tf.constant(-1)  # not used but needed for outputs

    # update online covariance and variance estimates

    # h = score function
    # aggregated over batch
    score_fun = optimizer.compute_gradients(tf.reduce_mean(sample_logprobs),
                                            tf.trainable_variables())

    # f = score function * reward
    rewarded_score_fun = optimizer.compute_gradients(
        tf.reduce_mean(sample_logprobs * -decoder.rewards),
        tf.trainable_variables())  # aggregated over batch

    # compute variance

    # (x_n - x_mean_n-1)
    cur_distance_from_mean = subtract_gradients(score_fun,
                                                decoder.score_fun_mean)
    # (x_n - x_mean_n-1)/n
    cur_distance_from_mean_norm = scale_gradients(cur_distance_from_mean,
                                                  1. / (tf.cast(
                                                      decoder.step,
                                                      tf.float32) + 1))
    # update mean of score function
    # x_mean_n = x_mean_n-1 + (x_n - x_mean_n-1)/n
    decoder.score_fun_mean = sum_gradients([decoder.score_fun_mean,
                                            cur_distance_from_mean_norm])
    # (x_n - x_mean_n)
    cur_distance_from_new_mean = subtract_gradients(score_fun,
                                                    decoder.score_fun_mean)
    # (x_n - x_mean_n_1)* (x_n - x_mean_n)
    cur_distance_from_mean_sq = multiply_gradients(
        [cur_distance_from_mean, cur_distance_from_new_mean])

    # update mean of distances from means
    # M_2,n = M_2,n-1 + (x_n - x_mean_n-1)*(x_n - x_mean_n)
    decoder.dist_mean = sum_gradients([decoder.dist_mean,
                                       cur_distance_from_mean_sq])
    # variance = M_2,n / (n-1)
    decoder.score_fun_variance = scale_gradients(decoder.dist_mean,
                                                 1. / (tf.cast(
                                                     decoder.step,
                                                     tf.float32)))

    # compute covariance

    # (y_n - y_mean_n-1)
    cur_distance_from_mean_2 = subtract_gradients(
        rewarded_score_fun, decoder.rewarded_score_fun_mean)
    cur_distance_from_mean_2_norm = scale_gradients(
        cur_distance_from_mean_2, 1. / (tf.cast(decoder.step, tf.float32) + 1))
    # update mean for rewarded score function
    decoder.rewarded_score_fun_mean = sum_gradients(
        [decoder.rewarded_score_fun_mean, cur_distance_from_mean_2_norm])

    # (x_n - x_mean_n) * (y_n - y_mean_n-1)
    means_product = multiply_gradients([cur_distance_from_mean_2,
                                        cur_distance_from_new_mean])
    # C_n = C_n-1 + (x_n - x_mean_n) * (y_n - y_mean_n-1)
    decoder.codist_mean = sum_gradients([decoder.codist_mean,
                                         means_product])
    # Cov = C_n/n
    decoder.covariance = scale_gradients(decoder.codist_mean,
                                         1. / (tf.cast(
                                             decoder.step, tf.float32) + 1))

    # scaling a = cov(f,h)/var(h)
    # for each component of the gradient
    scaling = divide_gradients(decoder.covariance, decoder.score_fun_variance)

    # scale score function
    # score_fun*-reward + score_fun*a
    scaled_score_fun = multiply_gradients([score_fun, scaling])
    full_grad = sum_gradients([rewarded_score_fun, scaled_score_fun])
    return BanditObjective(
        name="{} - expected_loss".format(decoder.name),
        decoder=decoder,
        samples=sample_ids,
        sample_logprobs=sample_logprobs,
        loss=tf.reduce_mean(tf.mul(tf.exp(sample_logprobs), -decoder.rewards),
                            [0, 1]),
        gradients=full_grad
    )


def cross_entropy_objective(decoder, optimizer, initial_temperature, clip_prob,
                            factor) -> BanditObjective:
    """Get bandit cross-entropy loss objective from decoder."""
    sample_ids, sample_logprobs, _ = _get_samples(decoder, neg=False)
    decoder.neg_sample_ix = tf.constant(-1)  # not used but needed for outputs

    # don't differentiate this
    scalars = tf.stop_gradient(
        ((decoder.rewards - decoder.baseline) -
         # entropy regularizer T*(log p +1)
         # T is annealed
         _get_temperature(
             initial_temperature,
             decoder.epoch)
         * (sample_logprobs + 1))
        # divide by factor * clipped sample prob
        / (factor * _clip_probs(tf.exp(sample_logprobs),
                                clip_prob)))
    gradients = optimizer.compute_gradients(
        tf.reduce_mean(-sample_logprobs * scalars))

    return BanditObjective(
        name="{} - cross-entropy".format(decoder.name),
        decoder=decoder,
        samples=sample_ids,
        sample_logprobs=sample_logprobs,
        loss=-tf.reduce_mean(tf.mul(sample_logprobs, decoder.rewards),
                             [0, 1]),
        gradients=gradients
    )


def pairwise_objective(decoder, optimizer, initial_temperature) -> \
        BanditObjective:
    """Get bandit loss objective from decoder."""
    sample_ids, sample_ids_2, sample_logprobs, sample_logprobs_2, neg_ix = \
        _get_sample_pairs_from_runtime_logits(decoder)
    pair_logprobs = (sample_logprobs + sample_logprobs_2)
    decoder.neg_sample_ix = neg_ix

    scalars = tf.stop_gradient(
        # loss from user feedback
        -(1 - (decoder.rewards - decoder.baseline)) +
        # entropy regularizer T*(log p +1)
        _get_temperature(initial_temperature,
                         decoder.epoch)
        * (pair_logprobs + 1))

    gradients = optimizer.compute_gradients(
        tf.reduce_mean(pair_logprobs * scalars),
        aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N)

    return BanditObjective(
        name="{} - pairwise".format(decoder.name),
        decoder=decoder,
        samples=[sample_ids, sample_ids_2],
        sample_logprobs=[sample_logprobs, sample_logprobs_2],
        loss=tf.reduce_mean(
            tf.mul(tf.exp(pair_logprobs), -(1 - decoder.rewards)),
            [0, 1]),
        gradients=gradients
    )


def pairwise_xent_objective(decoder, optimizer, initial_temperature,
                            clip_prob, factor) \
        -> BanditObjective:
    """Get bandit cross-entropy loss objective from decoder."""
    sample_ids, sample_ids_2, sample_logprobs, sample_logprobs_2, neg_ix = \
        _get_sample_pairs_from_runtime_logits(decoder)
    decoder.neg_sample_ix = neg_ix

    pair_logprobs = (sample_logprobs + sample_logprobs_2)
    pair_probs = tf.exp(pair_logprobs)

    scalars = tf.stop_gradient(
        (decoder.rewards -
         # entropy regularizer T*(log p +1)
         # T is annealed
         _get_temperature(
             initial_temperature, decoder.epoch)
         * (pair_logprobs + 1))
        /  # divide by factor * clipped sample prob
        (factor * _clip_probs(pair_probs, clip_prob)))

    gradients = optimizer.compute_gradients(
        tf.reduce_mean(-pair_logprobs * scalars))

    return BanditObjective(
        name="{} - pairwise_xent".format(decoder.name),
        decoder=decoder,
        samples=[sample_ids, sample_ids_2],
        sample_logprobs=[sample_logprobs,
                         sample_logprobs_2],
        loss=-tf.reduce_mean(tf.mul(pair_logprobs,
                                    decoder.rewards), [0, 1]),
        gradients=gradients
    )


def _get_temperature(initial_temperature, current_epoch):
    """
    Annealing temperature with decay function as in ACL paper:
    T = T0 / ((epoch + 1)^1/3)
    :param initial_temperature:
    :param current_epoch:
    :return:
    """
    return initial_temperature / (
        (tf.cast(current_epoch, tf.float32) + 1) ** 1 / 3.)


def _clip_probs(probs, prob_threshold):
    """ Clip probabilities to some threshold """
    if prob_threshold > 0.00:
        return tf.clip_by_value(probs, clip_value_min=prob_threshold,
                                clip_value_max=1)
    else:
        return probs


def _get_samples(decoder, neg=False):
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
        temperature=decoder.temperature,
        scope="{}/attention_decoder".format(decoder.name))

    # expansion is necessary for generalization to multiple samples
    # time x batch x sample_size
    sample_ids = tf.expand_dims(tf.pack(sample_ids), 2)
    # batch x sample_size
    sample_logprobs = tf.expand_dims(sample_logprob, 1)

    return sample_ids, sample_logprobs, neg_ix


def _get_sample_pairs(decoder):
    """ Sample a pair of outputs, one of them perturbed """
    sample_ids, sample_logprobs, _ = _get_samples(decoder, neg=False)
    sample_ids_neg, sample_logprobs_neg, neg_ix = \
        _get_samples(decoder, neg=True)
    return sample_ids, sample_ids_neg, sample_logprobs, \
           sample_logprobs_neg, neg_ix


def _get_sample_pairs_from_runtime_logits(decoder):
    """ Sample from runtime logits """
    sample_ids, sample_logprob, _ = \
        decoder.sample_from_runtime_logits(neg=False)
    sample_ids2, sample_logprob2, neg_ix = \
        decoder.sample_from_runtime_logits(neg=True)
    # time x batch x sample_size
    sample_ids = tf.expand_dims(tf.pack(sample_ids), 2)
    # batch x sample_size
    sample_logprobs = tf.expand_dims(sample_logprob, 1)
    # time x batch x sample_size
    sample_ids2 = tf.expand_dims(tf.pack(sample_ids2), 2)
    sample_logprobs2 = tf.expand_dims(sample_logprob2, 1)  # batch x sample_size
    return sample_ids, sample_ids2, sample_logprobs, sample_logprobs2, neg_ix


class ExploitOnlyTrainer(GenericBanditTrainer):
    """ Objective without sampling, greedily choosing best output """

    def __init__(self, decoders: List[Any], evaluator, l1_weight=0.,
                 l2_weight=0., initial_temperature=0., clip_norm=False,
                 optimizer=None, binary_feedback=False, store_gradients=False,
                 baseline=False, score_function=False) -> None:
        self.store_gradients = store_gradients
        if score_function:
            raise NotImplementedError("Score function control variate not "
                                      "implemented for ExploitOnlyObjective.")
        initial_temperature = initial_temperature
        objective = exploit_only_objective(
            decoders[0], optimizer, initial_temperature=initial_temperature)
        super(ExploitOnlyTrainer, self).__init__(
            objective, evaluator, l1_weight, l2_weight,
            clip_norm=clip_norm,
            optimizer=optimizer, pairwise=False,
            binary_feedback=binary_feedback, store_gradients=store_gradients,
            baseline=baseline)


class ExpectedLossTrainer(GenericBanditTrainer):
    """ EL objective with optional control variates (score function and BL) """

    def __init__(self, decoders: List[Any], evaluator, l1_weight=0.,
                 l2_weight=0., initial_temperature=0., clip_norm=False,
                 optimizer=None, binary_feedback=False, store_gradients=False,
                 baseline=False, score_function=False) -> None:
        initial_temperature = initial_temperature
        self.store_gradients = store_gradients
        if score_function:
            # score function control variate
            objective = expected_loss_objective_with_scorefun(decoders[0],
                                                              optimizer)
            if baseline:
                raise NotImplementedError(
                    "Score function and baseline control variate not "
                    "both usable for EL objective")
        else:
            objective = expected_loss_objective(decoders[0], optimizer,
                                                initial_temperature)
        super(ExpectedLossTrainer, self).__init__(
            objective, evaluator, l1_weight, l2_weight,
            clip_norm=clip_norm,
            optimizer=optimizer, pairwise=False,
            binary_feedback=binary_feedback,
            store_gradients=store_gradients, baseline=baseline)


class CrossEntropyTrainer(GenericBanditTrainer):
    """ CE objective with optional BL control variate """

    def __init__(self, decoders: List[Any], evaluator, l1_weight=0.,
                 l2_weight=0., initial_temperature=0., clip_norm=False,
                 optimizer=None, binary_feedback=False,
                 clip_prob=0.0, factor=1.0e10, store_gradients=False,
                 baseline=False, score_function=False) -> None:
        self.store_gradients = store_gradients
        if score_function:
            raise NotImplementedError("Score function control variate not "
                                      "implemented for CrossEntropyObjective.")
        objective = cross_entropy_objective(
            decoders[0], optimizer, initial_temperature=initial_temperature,
            clip_prob=clip_prob, factor=factor)
        super(CrossEntropyTrainer, self).__init__(
            objective, evaluator, l1_weight, l2_weight,
            clip_norm=clip_norm,
            optimizer=optimizer, pairwise=False,
            binary_feedback=binary_feedback, store_gradients=store_gradients,
            baseline=baseline)


class PairwiseTrainer(GenericBanditTrainer):
    """ PR-EL objective with pairwise sampling """
    def __init__(self, decoders: List[Any], evaluator, l1_weight=0.,
                 l2_weight=0., initial_temperature=0., clip_norm=False,
                 optimizer=None, binary_feedback=False, store_gradients=False,
                 baseline=False, score_function=False) -> None:
        self.store_gradients = store_gradients
        if score_function:
            raise NotImplementedError("Score function control variate not "
                                      "implemented for PR-EL-Objective.")
        objective = pairwise_objective(decoders[0], optimizer,
                                       initial_temperature=initial_temperature)
        super(PairwiseTrainer, self).__init__(
            objective, evaluator, l1_weight, l2_weight, clip_norm=clip_norm,
            optimizer=optimizer, pairwise=True, binary_feedback=binary_feedback,
            store_gradients=store_gradients, baseline=baseline)


class PairwiseXentTrainer(GenericBanditTrainer):
    """ PR-CE objective with pairwise sampling """
    def __init__(self, decoders: List[Any], evaluator, l1_weight=0.,
                 l2_weight=0., initial_temperature=0., clip_norm=False,
                 optimizer=None, binary_feedback=False,
                 clip_prob=0., factor=1.0e-10, store_gradients=False,
                 baseline=False, score_function=False) -> None:
        self.store_gradients = store_gradients
        if score_function:
            raise NotImplementedError("Score function control variate not "
                                      "implemented for PR-CE-Objective.")
        objective = pairwise_xent_objective(
            decoders[0], optimizer, initial_temperature=initial_temperature,
            clip_prob=clip_prob, factor=factor)
        super(PairwiseXentTrainer, self).__init__(
            objective, evaluator, l1_weight, l2_weight,
            clip_norm=clip_norm, optimizer=optimizer, pairwise=True,
            binary_feedback=binary_feedback, store_gradients=store_gradients,
            baseline=baseline)
