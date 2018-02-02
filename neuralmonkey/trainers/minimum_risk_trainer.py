"""Training objective for expected loss training."""

from typing import Callable, NamedTuple, Any

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

def expected_loss_objective(decoder: Decoder,
                            reward_function: RewardFunction,
                            number_of_samples: int = 5,
                            ce_smoothing: float = 0.,
                            alpha: float = 1.,
                            control_variate: str = None,
                            simulate_from_ref: bool = True,
                            service_url: str = None) -> Objective:
    """Minimum Risk Training with approximation over a sampled subspace

    'Minimum Risk Training for Neural Machine Translation'
    Details: http://www.aclweb.org/anthology/P16-1159

    :param decoder: a recurrent decoder to sample from
    :param reward_function: any evaluator object
    :param number_of_samples: number of samples for gradient approximation
    :param ce_smoothing: factor to smooth cross-entropy loss with
    :param alpha: determines the shape of the distribution, high: peaked
    :return: Objective object to be used in generic trainer
    """
    check_argument_types()

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
        rewards = [float(r) for r in
                   json.JSONDecoder().decode(response_dict)["predictions"]]
        return np.array(rewards, dtype=np.float32)

    # create empty TAs
    rewards=tf.TensorArray(dtype=tf.float32, size=number_of_samples, name="sample_rewards")
    logprobs=tf.TensorArray(dtype=tf.float32, size=number_of_samples, name="sample_logprobs")

    def body(index, rewards, logprobs) -> (int, tf.TensorArray, tf.TensorArray):

        sample_loop_result = decoder._decoding_loop(train_mode=False,
                                                   sample=True)
        sample_logits = sample_loop_result[0]
        sample_decoded = sample_loop_result[3]

        # rewards, shape (batch)
        if simulate_from_ref:
            # simulate from reference
            sample_reward = tf.py_func(_score_with_reward_function,
                                       [reference, sample_decoded], tf.float32)
        else:
            # retrieve from reward estimator model
            sample_sources = tf.transpose(
                decoder.encoders[0].input_sequence.inputs)
            sample_reward = tf.py_func(_get_reward_from_service, [sample_sources, sample_decoded], tf.float32)

        word_logprobs = -tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=sample_decoded, logits=sample_logits)

        # sum word log prob to sentence log prob
        # no masking here, since otherwise shorter sentences are preferred
        sent_logprobs = tf.reduce_sum(word_logprobs, axis=0)

        return (index+1,
                rewards.write(index, sample_reward),
                logprobs.write(index, sent_logprobs))

    condition = lambda i, r, p: i < number_of_samples
    _, final_rewards, final_logprobs = tf.while_loop(condition, body, (0, rewards, logprobs))

    samples_logprobs = final_logprobs.stack()  # samples, batch
    samples_reward = final_rewards.stack()  # samples, batch

    # if specified, compute the average reward baseline
    baseline = None

    reward_counter = tf.Variable(0.0, trainable=False,
                                 name="reward_counter")
    reward_sum = tf.Variable(0.0, trainable=False, name="reward_sum")

    if control_variate == "baseline":
        # increment the cumulative reward in the decoder
        reward_counter = tf.assign_add(reward_counter,
                                       tf.to_float(decoder.batch_size))
        # sum over batch, mean over samples
        reward_sum = tf.assign_add(reward_sum, tf.reduce_sum(tf.reduce_mean(samples_reward, axis=0)))
        baseline = tf.div(reward_sum,
                          tf.maximum(reward_counter, 1.0))
        samples_reward -= baseline

    renormalized_probs = tf.nn.softmax(samples_logprobs * alpha, dim=0)  # softmax over sample space
    scored_probs = -samples_reward * renormalized_probs
    total_loss = tf.reduce_sum(scored_probs, axis=0)

    # average over batch
    batch_loss = tf.reduce_mean(total_loss)
    if ce_smoothing > 0.0:
        batch_loss += tf.multiply(ce_smoothing, decoder.cost)

    tf.summary.scalar(
        "train_{}/self_mrt_cost".format(decoder.data_id),
        batch_loss,
        collections=["summary_train"])

    return Objective(
        name="{}_mrt".format(decoder.name),
        decoder=decoder,
        loss=batch_loss,
        gradients=None,
        weight=None
    )

