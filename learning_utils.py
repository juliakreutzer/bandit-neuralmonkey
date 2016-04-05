from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
from termcolor import colored
import regex as re
import os, time

def log(message, color='yellow'):
    print "{}: {}".format(colored(time.strftime("%Y-%m-%d %H:%M:%S"), color), message)


def print_header(title, args):
    """
    Prints the title of the experiment
    """
    print colored("".join("=" for _ in range(80)), 'green')
    print colored(title.upper(), 'green')
    print colored("".join("=" for _ in range(80)), 'green')

    print ""
    for arg in vars(args):
        value = str(getattr(args, arg))
        dots_count = 78 - len(arg) - len(value)
        # TODO if it is file print its path
        print "{} {} {}".format(arg, "".join(['.' for _ in range(dots_count)]), value)
    print ""

    os.system("echo last commit: `git log -1 --format=%H`")
    os.system("git --no-pager diff --color=always")
    print ""


def load_tokenized(text_file, lowercase=False):
    """
    Loads a tokenized text file a list of list of tokens.
    """

    return [re.split(ur"[ @#-]", l.rstrip()) for l in text_file]
    
    # if lowercase:
    #     return [ [ t.lower for token in sentence ] for sentence in tokenized ]
    # else:
    #     return lowercase
    



def load_char_based(test_file):
    """
    Loads a tokenized text for character-based decoding.
    """
    pass


def tokenize_char_seq(chars):
    return word_tokenize("".join(chars))


def corpus_bleu_deduplicated_unigrams(batch_sentences, decoded_sentences,
                                      weights, smoothing_function):

    deduplicated_sentences = []

    for sentence in decoded_sentences:

        last_w = None
        dedup_snt = []
    
        for word in sentence:
            if word != last_w:
                dedup_snt.append(word)
                last_w = word

        deduplicated_sentences.append(dedup_snt)

    return corpus_bleu(batch_sentences, deduplicated_sentences, weights,
                       smoothing_function)



def training_loop(sess, vocabulary, epochs, trainer,
                  decoder, train_feed_dicts, train_tgt_sentences,
                  val_feed_dict, val_tgt_sentences, char_based=False):
    """

    Performs the training loop for given graph and data.

    Args:

        sess: TF Session.

        vocabulary: Vocabulary used on the decoder side.

        epochs: Number of epochs for which the algoritm will learn.

        trainer: The trainer object.

        decoder: The decoder object.

        train_feed_dicts: List of feed dictionaires for training batches.

        train_tgt_sentences: List of batches of target training sentences for
            BLEU computation. Each sentence must be clsoed in one additional list
            (potentially more reference sentences for BLEU computation). Even if
            the character based decoding is done, these must be tokenized
            sentences.

        val_feed_dict: Feed dictionaty for the validation data.

        val_tgt_sentences: Validation target sentences for BLEU computation.
            Lists of lists (there may be multiple references for a sentece) of list
            of words. Each sentence must be clsoed in one additional list
            (potentially more reference sentences for BLEU computation). Even if
            the character based decoding is done, these must be tokenized
            sentences.

    """

    log("Starting training")
    step = 0
    bleu_smoothing = SmoothingFunction(epsilon=0.01).method1

    max_bleu = 0.0
    max_bleu_epoch = 0
    max_bleu_batch_no = 0

    for i in range(epochs):
        print ""
        log("Epoch {} starts".format(i + 1), color='red')

        for batch_n, (batch_feed_dict, batch_sentences) in \
                enumerate(zip(train_feed_dicts, train_tgt_sentences)):
            step += 1
            if step % 20 == 1:
                

                computation = trainer.run(sess, batch_feed_dict, batch_sentences, verbose=True)                

                decoded_sentences = \
                    vocabulary.vectors_to_sentences(computation[-decoder.max_output_len - 1:])

                if char_based:
                    decoded_sentences = \
                            [tokenize_char_seq(chars) for chars in decoded_sentences]

                bleu_1 = \
                    100 * corpus_bleu(batch_sentences, decoded_sentences,
                                      weights=[1., 0., 0., 0.],
                                      smoothing_function=bleu_smoothing)
                bleu_4 = \
                    100 * corpus_bleu(batch_sentences, decoded_sentences,
                                      weights=[0.25, 0.25, 0.25, 0.25],
                                      smoothing_function=bleu_smoothing)

                bleu_4_dedup = \
                    100 * corpus_bleu_deduplicated_unigrams(batch_sentences, decoded_sentences,
                                                            weights=[0.25, 0.25, 0.25, 0.25],
                                                            smoothing_function=bleu_smoothing)

                log("opt. loss: {:.4f}    dec. loss: {:.4f}    BLEU-1: {:.2f}    BLEU-4: {:.2f}    BLEU-4-dedup: {:.2f}"\
                        .format(computation[2], computation[1], bleu_1, bleu_4, bleu_4_dedup))
            else:
                trainer.run(sess, batch_feed_dict, batch_sentences, verbose=False)

            if step % 500 == 499:
                computation = sess.run([decoder.loss_with_decoded_ins, decoder.loss_with_gt_ins] \
                        + decoder.decoded_seq, feed_dict=val_feed_dict)
                decoded_val_sentences = \
                    vocabulary.vectors_to_sentences(computation[-decoder.max_output_len - 1:])

                if char_based:
                    decoded_val_sentences = \
                            [tokenize_char_seq(chars) for chars in decoded_val_sentences]

                val_bleu_1 = \
                        100 * corpus_bleu(val_tgt_sentences, decoded_val_sentences, weights=[1., 0., 0., 0.0],
                                          smoothing_function=bleu_smoothing)
                val_bleu_4 = \
                    100 * corpus_bleu(val_tgt_sentences, decoded_val_sentences, weights=[0.25, 0.25, 0.25, 0.25],
                                      smoothing_function=bleu_smoothing)

                val_bleu_4_dedup = \
                    100 * corpus_bleu_deduplicated_unigrams(val_tgt_sentences, decoded_val_sentences,
                                                            weights=[0.25, 0.25, 0.25, 0.25],
                                                            smoothing_function=bleu_smoothing)

                if val_bleu_4 > max_bleu:
                    max_bleu = val_bleu_4
                    max_bleu_epoch = i
                    max_bleu_batch_no = batch_n

                print ""
                log("Validation (epoch {}, batch number {}):".format(i, batch_n), color='cyan')
                log("opt. loss: {:.4f}    dec. loss: {:.4f}    BLEU-1: {:.2f}    BLEU-4: {:.2f}    BLEU-4-dedup: {:.2f}"\
                        .format(computation[1], computation[0], val_bleu_1, val_bleu_4, val_bleu_4_dedup), color='cyan')
                log("max BLEU-4 on validation: {:.2f} (in epoch {}, after batch number {})".format(max_bleu, max_bleu_epoch, max_bleu_batch_no), color='cyan')

                print ""
                print "Examples:"
                for sent, ref_sent in zip(decoded_val_sentences[:15], val_tgt_sentences):
                    print "    {}".format(" ".join(sent))
                    print colored("      ref.: {}".format(" ".join(ref_sent[0])), color="magenta")
                print ""
        
    log("Finished. Maximum BLEU-4 on validation data: {:.2f}, epoch {}".format(max_bleu, max_bleu_epoch))

