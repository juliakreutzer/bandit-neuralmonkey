# tests: lint, mypy

from collections import Counter
from typing import List, Tuple
import numpy as np

from neuralmonkey.evaluators.bleu import BLEUEvaluator


class GLEUEvaluator(object):
    """
    From "Google’s Neural Machine Translation System: Bridging the Gap
    between Human and Machine Translation" by Wu et al.
    (https://arxiv.org/pdf/1609.08144v2.pdf)
    "For the GLEU score, we record all sub-sequences of 1, 2, 3 or 4 tokens in
    output and target sequence (n-grams). We then compute a recall,
    which is the ratio of the number of matching n-grams to
    the number of total n-grams in the target (ground truth)
    sequence, and a precision, which is the ratio of
    the number of matching n-grams to the number of total
    n-grams in the generated output sequence. Then
    GLEU score is simply the minimum of recall and precision."

    Ngram counts are based on the bleu methods."""

    def __init__(self, n=4, deduplicate=False, name=None):
        self.n = n
        self.deduplicate = deduplicate
        self.BLEUEvaluator = BLEUEvaluator(n=4, deduplicate=False, name="BLEU")

        if name is not None:
            self.name = name
        else:
            self.name = "GLEU-{}".format(n)
            if self.deduplicate:
                self.name += "-dedup"

    def __call__(self, decoded, references):
        # type: (List[List[str]], List[List[str]]) -> float
        listed_references = [[s] for s in references]

        if self.deduplicate:
            decoded = self.BLEUEvaluator._deduplicate_sentences(decoded)

        return GLEUEvaluator.gleu(decoded, listed_references, self.n)


    @staticmethod
    def total_precision_recall(hypotheses: List[List[str]],
                                 references_list: List[List[List[str]]],
                                 ngrams: int,
                                 case_sensitive: bool) \
            -> Tuple[float, float]:
        """Computes the modified n-gram precision and recall
           on a list of sentences

        Arguments:
            hypothesis: List of output sentences as lists of words
            references: List of lists of reference sentences (as lists of
                words)
            n: n-gram order
            case_sensitive: Whether to perform case-sensitive computation
        """
        corpus_true_positives = 0
        corpus_generated_length = 0
        corpus_target_length = 0

        for n in range(1, ngrams+1):
            for hypothesis, references in zip(hypotheses, references_list):
                reference_counters = []

                for reference in references:
                    counter = BLEUEvaluator.ngram_counts(reference, n,
                                                         not case_sensitive)
                    reference_counters.append(counter)

                reference_counts = BLEUEvaluator.merge_max_counters(
                    reference_counters)
                corpus_target_length += sum(reference_counts.values())

                hypothesis_counts = BLEUEvaluator.ngram_counts(hypothesis, n,
                                                            not case_sensitive)
                true_positives = 0
                for ngram in hypothesis_counts:
                    true_positives += reference_counts[ngram]

                corpus_true_positives += true_positives
                corpus_generated_length += sum(hypothesis_counts.values())

            if corpus_generated_length == 0:
                return 0, 0

        return (corpus_true_positives / corpus_generated_length,
                corpus_true_positives / corpus_target_length)

    @staticmethod
    def gleu(hypotheses, references, ngrams=4, case_sensitive=True):
        # Type: (List[List[str]], List[List[List[str]]]) -> float
        """Computes GLEU on a corpus with multiple references. No smoothing.

        Arguments:
            hypotheses: List of hypotheses
            references: LIst of references. There can be more than one
                reference.
            ngram: Maximum order of n-grams. Default 4.
            case_sensitive: Perform case-sensitive computation. Default True.
        """
        prec, recall = GLEUEvaluator.total_precision_recall(
            hypotheses, references, ngrams, case_sensitive)

        return min(recall, prec)
