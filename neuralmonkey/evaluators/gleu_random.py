# tests: lint, mypy

from collections import Counter
from typing import List, Tuple
import numpy as np

from neuralmonkey.evaluators.bleu import BLEUEvaluator
from neuralmonkey.evaluators.gleu import GLEUEvaluator


class GLEURandomEvaluator(object):
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
    plus random Gaussian noise

    Ngram counts are based on the bleu methods."""

    def __init__(self, n=4, deduplicate=False, name=None):
        self.n = n
        self.deduplicate = deduplicate
        self.BLEUEvaluator = BLEUEvaluator(n=4, deduplicate=False, name="BLEU")

        if name is not None:
            self.name = name
        else:
            self.name = "GLEURandom-{}".format(n)
            if self.deduplicate:
                self.name += "-dedup"

    def __call__(self, decoded, references):
        # type: (List[List[str]], List[List[str]]) -> float
        listed_references = [[s] for s in references]

        if self.deduplicate:
            decoded = self.BLEUEvaluator._deduplicate_sentences(decoded)

        gleu = GLEUEvaluator.gleu(decoded, listed_references, self.n)

        # add random noise to gleu evaluation
        noise = np.random.normal(0.0, 1.0)

        # clip to original range
        noised = min(max(gleu+noise, 0.0), 1.0)

        return noised


