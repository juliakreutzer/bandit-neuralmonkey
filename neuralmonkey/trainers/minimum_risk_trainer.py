"""Training objective for expected loss training."""

from typing import Callable

import numpy as np
import tensorflow as tf
from typeguard import check_argument_types

from neuralmonkey.trainers.generic_trainer import Objective
from neuralmonkey.decoders.decoder import Decoder
from neuralmonkey.vocabulary import END_TOKEN, PAD_TOKEN


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
                            number_of_samples: int = 5,
                            ce_smoothing: float = 0.,
                            temperature: float = 1.) -> Objective:
    """Minimum Risk Training with approximation over a sampled subspace

    'Minimum Risk Training for Neural Machine Translation'
    Details: http://www.aclweb.org/anthology/P16-1159

    :param decoder: a recurrent decoder to sample from
    :param reward_function: any evaluator object
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

    samples_reward = []
    samples_logprobs = []

    # sample number_of_samples times, store logits, sample indices and rewards
    for sample_no in range(number_of_samples):
        # decoded, shape (time, batch)
        sample_loop_result = decoder.decoding_loop(train_mode=False, sample=True)
        sample_logits = sample_loop_result[0]
        sample_decoded = sample_loop_result[3]

        # rewards, shape (batch)
        sample_reward = tf.py_func(_score_with_reward_function,
                                   [reference, sample_decoded], tf.float32)

        word_logprobs = -tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=sample_decoded, logits=sample_logits)

        # sum word log prob to sentence log prob
        # no masking here, since otherwise shorter sentences are preferred
        sent_logprobs = tf.reduce_sum(word_logprobs, axis=0)

        samples_reward.append(sample_reward)
        samples_logprobs.append(sent_logprobs*temperature)

    # TODO make operations numerically stable

    # sum over samples for normalization
    Z = tf.reduce_logsumexp(tf.stack(samples_logprobs, axis=0), axis=0)
    renormalized_logprobs = [logprob - Z for logprob in samples_logprobs]

    renormalized_logprobs = tf.Print(renormalized_logprobs, [Z, samples_logprobs, renormalized_logprobs], "Z, sample logprobs, renormalized", summarize=10)

    total_loss = tf.zeros((decoder.batch_size,))
    # iterate over samples again to compute loss
    for sample_no in range(number_of_samples):

        # REINFORCE gradient, shape (batch)
        score = tf.stop_gradient(tf.negative(samples_reward[sample_no])) * \
                renormalized_logprobs[sample_no]

        # vector of batch length
        total_loss += score

    # average over samples
    total_loss /= tf.to_float(number_of_samples)

    # TODO smooth with xent

    # average over batch
    batch_loss = tf.reduce_mean(total_loss)
    batch_loss = tf.Print(batch_loss, [batch_loss], "batch_loss")

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
