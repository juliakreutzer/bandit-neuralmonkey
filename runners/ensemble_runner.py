import tensorflow as tf


class EnsembleRunner(object):

    def __init__(self, graphs, batch_size):

        self.decoders = decoders
        self.batch_size = batch_size
        # TODO check vocabularies are the same
        self.vocabulary = decoders[0].vocabulary
        # TODO check data ids are same
        self.data_id = decoders[0].data_id

        # TODO constructor could just be called with vocabulary and data series
        # id



    def __call__(self, sess, dataset, coders, loss_tensors, decoded_tensors):

        batched_dataset = dataset.batch_dataset(self.batch_size)
        decoded_sentences = []

        batch_count = 0

        for batch in batched_dataset:
            batch_feed_dict = feed_dicts(batch, coders, train=False)
            batch_count += 1

            # decoder x batch
            decoder_probs = []
            # decoder x time x batch
            decoder_seqs = []

            for decoder in self.decoders:

                outputs = [#decoder.loss_with_gt_ins,
                           #decoder.loss_with_decoded_ins,
                           decoder.logprob_sequence]

                computation = sess.run(outputs + decoder.decoded_seq)

                #loss_gt = computation[0]
                #loss_decoded = computation[1]

                # batch
                batch_log_probabilities = coputation[0]
                decoder_probs.append(batch_log_probabilities)

                # time x batch
                decoded = computation[1:]
                decoder_seqs.append(decoded)

            # batch -> range(decoders)
            best_decoders = np.argmax(decoder_probs, 0)

            # vysledek je time x batch tak aby se to veslo do
            # vocabulary.vectors_to_sentences

            # chci udelat choose z decoder x time x batch a mam vektor
            # velikosti batch co ma hodnoty z range(decoders) takze
            # vlastne neni potreba delat zadny transpose
            # vysledek je time x batch
            ensemble_winners = np.choose(best_decoders, decoder_seqs)

            # pridam batch vet
            decoded_sentences += self.vocabulary.vectors_to_sentences(
                ensemble_winners)

        return decoded_sentences


# len vec je mat.shape[-1]
# hodnoty vec jsou range(mat.shape[0])
