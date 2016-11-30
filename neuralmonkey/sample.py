import tensorflow as tf
import numpy as np

from neuralmonkey.logging import log, debug
from neuralmonkey.encoders.sentence_encoder import SentenceEncoder
from neuralmonkey.hypothesis import Hypothesis
from neuralmonkey.tf_manager import _feed_dicts

# TODO implement sampling as described in MinRisk paper

START_TOKEN_INDEX = 1
END_TOKEN_INDEX = 2

class SampleRunner(object):
    def __init__(self, decoder):
        """Construct a new instance of the runner.

        Arguments:
            decoder: The decoder to use for decoding
        """
        self.decoder = decoder
        self.vocabulary = decoder.vocabulary
        self.sampler = Sampler(decoder=decoder)

    def __call__(self, sess, dataset, coders, extra_fetches=None):
        sentence_datasets = dataset.batch_dataset(1)
        decoded_sentences = []

        train_loss = 0.0
        runtime_loss = 0.0
        sentence_count = 0

        for sentence_ds in sentence_datasets:
            sentence_count += 1
            feed_dict = _feed_dicts(sentence_ds, coders, train=False)
            # TODO so far only one sample
            sampled_hyp, sampled_text = self.sampler._sample(sess, sentence_ds, coders, 1, unique=False)[0]
            decoded_sentences.append(sampled_text)

            # Now, compute the loss for the best hypothesis (if we have the
            # target data)
            if dataset.has_series(self.decoder.data_id):
                losses = {
                    "train_loss": self.decoder.train_loss,
                    "runtime_loss": self.decoder.runtime_loss}

                # The feed dict for losses needs train and runtime logits, train
                # targets and padding weights. The targets and paddings are in
                # the feed dict constructed in the beginning of this function.
                loss_feed_dict = {
                    tensor: np.expand_dims(state, 0) for tensor, state in zip(
                    self.decoder.runtime_logits, sampled_hyp.runtime_logits)}

                loss_feed_dict.update(feed_dict)
                loss_comp = sess.run(losses, feed_dict=loss_feed_dict)

                train_loss += loss_comp["train_loss"]
                runtime_loss += loss_comp["runtime_loss"]

                # The train and runtime loss is an average over the whole dataset
        train_loss /= sentence_count
        runtime_loss /= sentence_count

        return decoded_sentences, train_loss, runtime_loss, None  # only return sentences


class Sampler(object):
    def __init__(self, decoder):
        self.decoder = decoder
        self.max_len = decoder.max_output
        self.vocabulary = decoder.vocabulary

    def _sample(self, sess, sentence, coders, sample_size, unique=True):  # TODO could also implement runner interface
        """
        Sample from the model distribution given an input
        :param sentence: input sentence to translate (in beam search: sentence_ds)
        :param coders: list of en- and decoders
        :param sample_size: number of samples
        :param unique: wether to check the sampled outputs for duplicates
        :return:
        """

        samples = []  # we don't include gold standard since we don't assume it exists
        feed_dict = _feed_dicts(sentence, coders, train=False)

        i = 0
        while i < sample_size:
            n = 0
            fetches = [self.decoder.runtime_rnn_states[0]]  # fetch rnn states
            fetches += [e.encoded for e in coders
                        if isinstance(e, SentenceEncoder)]  # fetch encoded

            for encoder in coders:
                if isinstance(encoder, SentenceEncoder):
                    fetches += encoder.outputs_bidi_t  # fetch output of encoder

            computation = sess.run(fetches, feed_dict=feed_dict)  # get fetches

            # Use the fetched values to create the continuation feed dict
            init_feed_dict = {tensor: value for tensor, value in
                              zip(fetches, computation)}

            initial_state = computation[0]
            hyp = Hypothesis([START_TOKEN_INDEX], 0.0, initial_state)

            while n < self.max_len:
                # sample n-th target word
                word_prob, word_id = self.decoder.sample_singleton(1, n) # TODO improve for batch

                # Fetch the next state and rnn output
                # Note that there is one more states than outputs because
                # the initial state is included in the list.
                s_fetches = {
                    "state": self.decoder.runtime_rnn_states[n + 1],
                    "output": self.decoder.runtime_rnn_outputs[n],
                    "sample_val": word_prob,
                    "sample_ids": word_id,
                    "logits": self.decoder.runtime_logits[n]
                }

                # For computation of the next step we use the feed dict
                # created in the last time step and add the output from the
                # encoder.
                s_feed_dict = {
                    self.decoder.runtime_rnn_states[n]:
                        np.expand_dims(hyp.state, 0)}

                s_feed_dict.update(init_feed_dict)
                s_feed_dict.update(feed_dict)

                if n > 0:
                    s_feed_dict[
                        self.decoder.runtime_rnn_outputs[n - 1]
                    ] = np.expand_dims(hyp.rnn_output, 0)

                res = sess.run(s_fetches, s_feed_dict)

                # Shapes and description of the tensors from "res":
                # state:       (1, rnn_size)  [intermediate RNN state]
                # output:      (1, rnn_size)  [projected cell_output]
                # topk_values: (1, beam_size) [logprobs of words]
                # topk_ids:    (1, beam_size) [indices to vocab]
                # logits:      (1, vocabulary_sice) [logits to vocab]
                state = res["state"][0]
                output = res["output"][0]
                val = res["sample_val"]
                index = res["sample_ids"]
                logits = res["logits"][0]

                #debug("Next sampled word is '{}' ({}) with log prob {}".format(
                #    self.decoder.vocabulary.index_to_word[index], index, val), color='blue')

                # add to y
                hyp = hyp.extend(index, val, state, output, logits)
                # if y_t == EOS then break
                if hyp.latest_token == END_TOKEN_INDEX:
                    break
                n += 1

            hyp_text = " ".join(self.decoder.vocabulary.hypothesis_to_sentence(hyp)[1:])  # cut off '<s>'

            if unique:  # only add to samples if not already in sample set
                seen_before = any(sample == hyp for sample in samples)
                if not seen_before:
                    i += 1
                    samples.append((hyp, hyp_text))
                    debug("Sampled hypothesis {}".format(hyp_text))
                else:
                    debug("Sampled duplicate: {}".format(hyp_text))
            else:
                i += 1
                samples.append((hyp, hyp_text))
                debug("Sampled hypothesis '{}' with log prob {} / prob {}".format(
                    hyp_text, hyp.log_prob, np.exp(hyp.log_prob)))

        return samples






