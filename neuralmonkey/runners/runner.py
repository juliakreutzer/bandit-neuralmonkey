import tensorflow as tf

from neuralmonkey.learning_utils import feed_dicts
from neuralmonkey.logging import debug

# tests: lint, mypy

# pylint: disable=too-few-public-methods
class GreedyRunner(object):
    def __init__(self, decoder, batch_size):
        self.decoder = decoder
        self.batch_size = batch_size
        self.vocabulary = decoder.vocabulary

    def __call__(self, sess, dataset, coders, get_probs=False):
        batched_dataset = dataset.batch_dataset(self.batch_size)
        decoded_sentences = []
        if dataset.has_series(self.decoder.data_id):
            losses = [self.decoder.train_loss,
                      self.decoder.runtime_loss]
        else:
            losses = [tf.zeros([]), tf.zeros([])]

        loss_with_gt_ins = 0.0
        loss_with_decoded_ins = 0.0
        batch_count = 0
        for batch in batched_dataset:
            batch_feed_dict = feed_dicts(batch, coders, train=False)
            batch_count += 1
            fetch = []

            # if are are target sentence, we will compute also the losses
            if dataset.has_series(self.decoder.data_id):
                fetch = [self.decoder.train_loss,
                         self.decoder.runtime_loss]

            if get_probs:
                prob_index = len(fetch)
                fetch.append(self.decoder.decoded_logprobs)

            decoded_start_index = len(fetch)
            fetch += self.decoder.decoded

            computation = sess.run(fetch, feed_dict=batch_feed_dict)

            if dataset.has_series(self.decoder.data_id):
                loss_with_gt_ins += computation[0]
                loss_with_decoded_ins += computation[1]

            if get_probs:
                logprobs = computation[prob_index]
                debug("log probability of the first sentence in batch: {}".
                      format(logprobs[0]), label="dumper")

            decoded_sentences_batch = self.vocabulary.vectors_to_sentences(
                computation[decoded_start_index:])

            decoded_sentences += decoded_sentences_batch

        return decoded_sentences, \
               loss_with_gt_ins / batch_count, \
               loss_with_decoded_ins / batch_count
