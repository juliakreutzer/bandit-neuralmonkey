"""Training objective for expected loss training."""

from typing import Callable

import numpy as np
import tensorflow as tf
from typeguard import check_argument_types

from neuralmonkey.trainers.generic_trainer import Objective
from neuralmonkey.decoders.decoder import Decoder
from neuralmonkey.vocabulary import END_TOKEN, PAD_TOKEN

import requests
import json

# pylint: disable=invalid-name
RewardFunction = Callable[[np.ndarray, np.ndarray], np.ndarray]
# pylint: enable=invalid-name


def reinforce_score(reward: tf.Tensor,
                    baseline: tf.Tensor,
                    decoded: tf.Tensor,
                    logits: tf.Tensor) -> tf.Tensor:
    """Cost function whose derivative is the REINFORCE equation.

    This implements the primitive function to the central equation of the
    REINFORCE algorithm that estimates the gradients of the loss with respect
    to decoder logits.

    The second term of the product is the derivative of the log likelihood of
    the decoded word. The reward function and the optional baseline are however
    treated as a constant, so they influence the derivate
    only multiplicatively.

    :param reward: reward for the selected sample
    :param baseline: baseline to subtract from the reward
    :param decoded: token indices for sampled translation
    :param logits: logits for sampled translation
    :param mask: 1 if inside sentence, 0 if outside
    :return:
    """
    # shape (batch)
    if baseline is not None:
        reward -= baseline

    # runtime probabilities, shape (time, batch, vocab)
    # pylint: disable=invalid-unary-operand-type
    word_logprobs = -tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=decoded, logits=logits)

    # sum word log prob to sentence log prob
    # no masking here, since otherwise shorter sentences are preferred
    sent_logprobs = tf.reduce_sum(word_logprobs, axis=0)

    # REINFORCE gradient, shape (batch)
    score = tf.stop_gradient(tf.negative(reward)) * sent_logprobs
    return score


def expected_loss_objective(decoder: Decoder,
                            reward_function: RewardFunction,
                            control_variate: str = None,
                            simulate_from_ref: bool = True,
                            service_url: str=None) -> Objective:
    """Construct Expected Loss objective for training with bandit feedback.

    'Bandit Structured Prediction for Neural Sequence-to-Sequence Learning'
    Details: http://www.aclweb.org/anthology/P17-1138

    :param decoder: a recurrent decoder to sample from
    :param reward_function: any evaluator object
    :param control_variate: optional 'baseline' average reward
    :return: Objective object to be used in generic trainer
    """
    check_argument_types()

    # decoded, shape (time, batch)
    # pylint: disable=protected-access
    sample_loop_result = decoder._decoding_loop(train_mode=False, sample=True)
    sample_logits = sample_loop_result[0]
    sample_decoded = sample_loop_result[3]

    reference = decoder.train_inputs

    def _score_with_reward_function(references: np.array,
                                    hypotheses: np.array) -> np.array:
        """Score (time, batch) arrays with sentence-based reward function.

        Parts of the sentence after generated <pad> or </s> are ignored.
        BPE-postprocessing is also included.

        :param references: array of indices of references, shape (time, batch)
        :param hypotheses: array of indices of hypotheses, shape (time, batch)
        :return: an array of batch length with float rewards
        """
        rewards = []
        for refs, hyps in zip(references.transpose(), hypotheses.transpose()):
            ref_seq = []
            hyp_seq = []
            for r_token in refs:
                token = decoder.vocabulary.index_to_word[r_token]
                if token == END_TOKEN or token == PAD_TOKEN:
                    break
                ref_seq.append(token)
            for h_token in hyps:
                token = decoder.vocabulary.index_to_word[h_token]
                if token == END_TOKEN or token == PAD_TOKEN:
                    break
                hyp_seq.append(token)
            # join BPEs, split on " " to prepare list for evaluator
            refs_tokens = " ".join(ref_seq).replace("@@ ", "").split(" ")
            hyps_tokens = " ".join(hyp_seq).replace("@@ ", "").split(" ")
            reward = float(reward_function([hyps_tokens], [refs_tokens]))
            rewards.append(reward)
        return np.array(rewards, dtype=np.float32)

    def _get_reward_from_service(sources: np.array, hypotheses: np.array) -> np.array:
        """Request the reward for a (time, batch) array from service.

        :param sources: array of indices of sources, shape (time, batch)
        :param hypotheses: array of indices of hypotheses, shape (time, batch)
        :return: an array of batch length with float rewards
        """
        request_inputs = []
        for srcs, hyps in zip(sources.transpose(), hypotheses.transpose()):
            hyp_seq = []
            src_seq = []
            for h_token in hyps:
                token = decoder.vocabulary.index_to_word[h_token]
                if token == END_TOKEN or token == PAD_TOKEN:
                    break
                hyp_seq.append(token)
            for s_token in srcs:
                token = decoder.encoders[0].vocabulary.index_to_word[s_token]
                if token == END_TOKEN or token == PAD_TOKEN:
                    break
                src_seq.append(token)
            request_inputs.append((" ".join(src_seq), " ".join(hyp_seq)))
        # request feedback
        url = service_url
        data = {"inputs": request_inputs}
        headers = {'content-type': 'application/json'}

        response = requests.post(url, data=json.dumps(data),
                                 headers=headers)

        response_dict = response.content.decode()
        rewards = [float(r) for r in json.JSONDecoder().decode(response_dict)["predictions"]]
        return np.array(rewards, dtype=np.float32)


    # rewards, shape (batch)
    if simulate_from_ref:
        sample_reward = tf.py_func(_score_with_reward_function,
                                   [reference, sample_decoded], tf.float32)
    else:
        sample_sources = tf.transpose(decoder.encoders[0].input_sequence.inputs)
        sample_reward = tf.py_func(_get_reward_from_service,
                                   [sample_sources, sample_decoded], tf.float32)

    # if specified, compute the average reward baseline
    baseline = None

    reward_counter = tf.Variable(0.0, trainable=False,
                                 name="reward_counter")
    reward_sum = tf.Variable(0.0, trainable=False, name="reward_sum")

    if control_variate == "baseline":
        # increment the cumulative reward in the decoder
        reward_counter = tf.assign_add(reward_counter,
                                       tf.to_float(decoder.batch_size))
        reward_sum = tf.assign_add(reward_sum, tf.reduce_sum(sample_reward))
        baseline = tf.div(reward_sum,
                          tf.maximum(reward_counter, 1.0))

    tf.summary.scalar(
        "sample_{}/reward".format(decoder.data_id),
        tf.reduce_mean(sample_reward),
        collections=["summary_train"])

    # REINFORCE score: shape (time, batch, vocab)
    sent_loss = reinforce_score(
        sample_reward, baseline, sample_decoded, sample_logits)

    batch_loss = tf.reduce_mean(sent_loss)

    tf.summary.scalar(
        "train_{}/self_bandit_cost".format(decoder.data_id),
        batch_loss,
        collections=["summary_train"])

    return Objective(
        name="{}_bandit".format(decoder.name),
        decoder=decoder,
        loss=batch_loss,
        gradients=None,
        weight=None
    )


def dpm_objective(decoder:Decoder, control_variate: str=None) -> Objective:
    """ Deterministic Propensity Matching

    See: http://www.aclweb.org/anthology/D/D17/D17-1272.pdf

    control variates: reweighting and baseline
    """

    check_argument_types()

    # logged translation
    hypothesis = decoder.train_inputs  # time, batch

    # TODO might normalize by length
    hyp_logprobs = -tf.contrib.seq2seq.sequence_loss(
        tf.transpose(decoder.train_logits, perm=[1,0,2]),  # batch length voc -> before: length, batch, voc
        tf.transpose(hypothesis), tf.transpose(decoder.train_padding),
        average_across_batch=False, average_across_timesteps=True) # shape afterwards -> batch
    hyp_logprobs = tf.Print(hyp_logprobs, [hyp_logprobs], "logprobs", summarize=10)

    # weigh model probabilities by reward
    rewards = decoder.train_rewards
    rewards = tf.Print(rewards, [rewards], "rewards", summarize=10)

    hyp_probs = tf.exp(hyp_logprobs)
    hyp_probs = tf.Print(hyp_probs, [hyp_probs], "probs", summarize=10)

    if control_variate == "reweighting":
        hyp_probs /= tf.reduce_sum(hyp_probs)
        # TODO baseline?

    loss = tf.reduce_sum(-rewards*hyp_probs)

    loss = tf.Print(loss, [loss], "loss", summarize=10)

    return Objective(
        name="{}_dpm".format(decoder.name),
        decoder=decoder,
        loss=loss,
        gradients=None,
        weight=None
    )

