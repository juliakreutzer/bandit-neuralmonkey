"""Training objectives for reinforcement learning."""

from typing import Callable

import numpy as np
import tensorflow as tf
from typeguard import check_argument_types

from neuralmonkey.trainers.generic_trainer import Objective
from neuralmonkey.decoders.decoder import Decoder
from neuralmonkey.vocabulary import END_TOKEN, PAD_TOKEN, Vocabulary
from neuralmonkey.dataset.buffer import TrainingBuffer, WeightedTrainInstance

# pylint: disable=invalid-name
RewardFunction = Callable[[np.ndarray, np.ndarray], np.ndarray]
# pylint: enable=invalid-name


def wxent_objective(decoder: Decoder):
    # neg log likelihood * reward * prob_current/logprob_historic
    # factors are constant wrt model parameters, just for importance sampling
    # TODO add baseline or normalization
    instance_weight = tf.multiply(decoder.reward,
                                  tf.exp(-decoder.train_xents
                                         -decoder.historic_logprob))
    total_loss = tf.multiply(-decoder.train_xents,  # - to revert - in NLL
                             tf.stop_gradient(-instance_weight))  # since descent

    total_loss = tf.Print(total_loss, ["current logprob", -decoder.train_xents,
                                       "historic logprob", decoder.historic_logprob,
                                       "importance",
                                       tf.exp(-decoder.train_xents
                                                 -decoder.historic_logprob),
                                       "reward", decoder.reward,
                                       "final weight:", instance_weight])

    batch_loss = tf.reduce_mean(total_loss)

    return Objective(
        name="{}_wxent".format(decoder.name),
        decoder=decoder,
        loss=batch_loss,
        gradients=None,
        weight=None
    )


# pylint: disable=too-many-locals
def rl_objective(decoder: Decoder,
                 src_vocabulary: Vocabulary,
                 trg_vocabulary: Vocabulary,
                 buffer: TrainingBuffer,
                 reward_function: RewardFunction,
                 subtract_baseline: bool = False,
                 normalize: bool = False,
                 temperature: float = 1.,
                 ce_smoothing: float = 0.,
                 alpha: float = 1.,
                 sample_size: int = 1) -> Objective:
    """Construct RL objective for training with sentence-level feedback.

    Depending on the options the objective corresponds to:
    1) sample_size = 1, normalize = False, ce_smoothing = 0.0
     Bandit objective (Eq. 2) described in 'Bandit Structured Prediction for
     Neural Sequence-to-Sequence Learning'
     (http://www.aclweb.org/anthology/P17-1138)
     It's recommended to set subtract_baseline = True.
    2) sample_size > 1, normalize = True, ce_smoothing = 0.0
     Minimum Risk Training as described in 'Minimum Risk Training for Neural
     Machine Translation' (http://www.aclweb.org/anthology/P16-1159) (Eq. 12).
    3) sample_size > 1, normalize = False, ce_smoothing = 0.0
     The Google 'Reinforce' objective as proposed in 'Google’s NMT System:
     Bridging the Gap between Human and Machine Translation'
     (https://arxiv.org/pdf/1609.08144.pdf) (Eq. 8).
    4) sample_size > 1, normalize = False, ce_smoothing > 0.0
     Google's 'Mixed' objective in the above paper (Eq. 9),
     where ce_smoothing implements alpha.

    Note that 'alpha' controls the sharpness of the normalized distribution,
    while 'temperature' controls the sharpness during sampling.

    :param decoder: a recurrent decoder to sample from
    :param reward_function: any evaluator object
    :param subtract_baseline: avg reward is subtracted from obtained reward
    :param normalize: the probabilities of the samples are re-normalized
    :param sample_size: number of samples to obtain feedback for
    :param ce_smoothing: add cross-entropy loss with this coefficient to loss
    :param alpha: determines the shape of the normalized distribution
    :param temperature: the softmax temperature for sampling
    :return: Objective object to be used in generic trainer
    """
    check_argument_types()

    reference = decoder.train_inputs
    # add source for logging (hack)
    source = decoder.encoders[0].input_sequence.inputs  # batch x time

    def _score_with_reward_function_and_log(sources: np.array,
                                    references: np.array,
                                    hypotheses: np.array,
                                    logprobs: np.array) -> np.array:
        """Score (time, batch) arrays with sentence-based reward function.

        Parts of the sentence after generated <pad> or </s> are ignored.
        BPE-postprocessing is also included.

        :param sources: array of indices of sources, shape (batch, time)
        :param references: array of indices of references, shape (time, batch)
        :param hypotheses: array of indices of hypotheses, shape (time, batch)
        :param logprobs: array of log probabilities of hypotheses, shape (batch)
        :return: an array of batch length with float rewards
        """
        rewards = []
        for srcs, refs, hyps, logprob in zip(sources, references.transpose(), hypotheses.transpose(), logprobs.transpose()):
            src_seq = []
            ref_seq = []
            hyp_seq = []
            # TODO decode sources
            for r_token in refs:
                token = trg_vocabulary.index_to_word[r_token]
                if token == END_TOKEN or token == PAD_TOKEN:
                    break
                ref_seq.append(token)
            for h_token in hyps:
                token = trg_vocabulary.index_to_word[h_token]
                if token == END_TOKEN or token == PAD_TOKEN:
                    break
                hyp_seq.append(token)
            for s_token in srcs:
                token = src_vocabulary.index_to_word[s_token]
                if token == END_TOKEN or token == PAD_TOKEN:
                    break
                src_seq.append(token)
            # join BPEs, split on " " to prepare list for evaluator
            refs_tokens = " ".join(ref_seq).replace("@@ ", "").split(" ")
            hyps_tokens = " ".join(hyp_seq).replace("@@ ", "").split(" ")
            src_tokens = " ".join(src_seq).replace("@@ ", "").split(" ")

            reward = float(reward_function([hyps_tokens], [refs_tokens]))

            if reward > 0:
                # TODO also log pseudo-references and use them for supervised updates (when reward e.g. > 0.9) *without* weighting (doesn't matter how likely it was in old model)
                # TODO and add to training set? keep in memory for x iterations? keep in id space? make buffer object that has average reward/values available
                #print(" ".join(hyps_tokens), " ".join(src_tokens), " ".join(refs_tokens), np.exp(logprob), reward, reward/np.exp(logprob), np.log(reward), logprob)
                # no BPE postprocessing or the like
                train_instance = WeightedTrainInstance(src=src_seq, trg=hyp_seq,
                                                       reward=reward, logprob=logprob)
                buffer.add_single(train_instance)

            rewards.append(reward)
        return np.array(rewards, dtype=np.float32)

    samples_rewards = []
    samples_logprobs = []

    for _ in range(sample_size):
        # sample from logits
        # decoded, shape (time, batch)
        sample_loop_result = decoder.decoding_loop(train_mode=False,
                                                   sample=True,
                                                   temperature=temperature)
        sample_logits = sample_loop_result[0]
        sample_decoded = sample_loop_result[3]

        # pylint: disable=invalid-unary-operand-type
        word_logprobs = -tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=sample_decoded, logits=sample_logits)

        # sum word log prob to sentence log prob
        # no masking here, since otherwise shorter sentences are preferred
        sent_logprobs = tf.reduce_sum(word_logprobs, axis=0)

        # rewards, shape (batch)
        # simulate from reference
        sample_reward = tf.py_func(_score_with_reward_function_and_log,
                                   [source, reference, sample_decoded, sent_logprobs],
                                   tf.float32)


        # inspect samples, rewards and probs for debugging
        #sent_logprobs = tf.Print(sent_logprobs, [tf.exp(sent_logprobs), sample_reward, sample_reward/tf.exp(sent_logprobs)])

        samples_rewards.append(sample_reward)   # sample_size x batch
        samples_logprobs.append(sent_logprobs)  # sample_size x batch

    # stack samples, sample_size x batch
    samples_rewards_stacked = tf.stack(samples_rewards)
    samples_logprobs_stacked = tf.stack(samples_logprobs)

    if subtract_baseline:
        # if specified, compute the average reward baseline
        reward_counter = tf.Variable(0.0, trainable=False,
                                     name="reward_counter")
        reward_sum = tf.Variable(0.0, trainable=False, name="reward_sum")
        # increment the cumulative reward
        reward_counter = tf.assign_add(
            reward_counter, tf.to_float(decoder.batch_size * sample_size))
        # sum over batch and samples
        reward_sum = tf.assign_add(reward_sum,
                                   tf.reduce_sum(samples_rewards_stacked))
        # compute baseline: avg of previous rewards
        baseline = tf.div(reward_sum,
                          tf.maximum(reward_counter, 1.0))
        samples_rewards_stacked -= baseline

        tf.summary.scalar(
            "train_{}/rl_reward_baseline".format(decoder.data_id),
            tf.reduce_mean(baseline), collections=["summary_train"])

    if normalize:
        # normalize over sample space
        samples_logprobs_stacked = tf.nn.softmax(
            samples_logprobs_stacked * alpha, dim=0)

    scored_probs = tf.stop_gradient(
        tf.negative(samples_rewards_stacked)) * samples_logprobs_stacked

    # sum over samples
    total_loss = tf.reduce_sum(scored_probs, axis=0)

    # average over batch
    batch_loss = tf.reduce_mean(total_loss)

    if ce_smoothing > 0.0:
        batch_loss += tf.multiply(ce_smoothing, decoder.cost)

    tf.summary.scalar(
        "train_{}/self_rl_cost".format(decoder.data_id),
        batch_loss,
        collections=["summary_train"])

    return Objective(
        name="{}_rl".format(decoder.name),
        decoder=decoder,
        loss=batch_loss,
        gradients=None,
        weight=None
    )
