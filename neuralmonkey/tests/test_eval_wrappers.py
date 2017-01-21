# test evaluation metric wrappers

import unittest

from neuralmonkey.evaluators.multeval import MultEvalWrapper
from neuralmonkey.evaluators.beer import BeerWrapper
from neuralmonkey.evaluators.bleu import BLEUEvaluator

ref = "I like tulips ."
hyp = "I hate flowers and stones ."

class TestExternalEvaluators(unittest.TestCase):

    def test_multeval_bleu(self):
        multeval = MultEvalWrapper("scripts/multeval-0.5.1/multeval.sh",
                                   metric="bleu")
        bleu = multeval([hyp], [ref])
        print("MultEval BLEU: {}".format(bleu))
        max_bleu = multeval([ref], [ref],)
        print("MultEval BLEU max: {}".format(max_bleu))
        min_bleu = multeval([], [ref])
        print("MultEval BLEU min: {}".format(min_bleu))
        self.assertGreater(max_bleu, min_bleu)
        self.assertGreater(bleu, min_bleu)
        self.assertGreaterEqual(max_bleu, bleu)

    def test_multeval_ter(self):
        multeval = MultEvalWrapper("scripts/multeval-0.5.1/multeval.sh",
                                   metric="ter")
        ter = multeval([hyp], [ref])
        print("MultEval TER: {}".format(ter))
        min_ter = multeval([ref], [ref])
        print("MultEval TER min: {}".format(min_ter))
        max_ter = multeval([], [ref])
        print("MultEval TER max: {}".format(max_ter))
        self.assertGreater(max_ter, min_ter)
        self.assertGreater(ter, min_ter)
        # does not hold for BEER
        #self.assertGreaterEqual(max_ter, ter)

    def test_multeval_meteor(self):
        multeval = MultEvalWrapper("scripts/multeval-0.5.1/multeval.sh",
                                   metric="meteor")
        meteor = multeval([hyp], [ref])
        print("MultEval METEOR: {}".format(meteor))
        max_meteor = multeval([ref], [ref])
        print("MultEval METEOR max: {}".format(max_meteor))
        min_meteor = multeval([], [ref])
        print("MultEval METEOR min: {}".format(min_meteor))
        self.assertGreater(max_meteor, min_meteor)
        self.assertGreater(meteor, min_meteor)
        self.assertGreaterEqual(max_meteor, meteor)

    def test_bleu(self):
        bleu_evaluator = BLEUEvaluator()
        bleu = bleu_evaluator([hyp], [ref])
        print("NM BLEU: {}".format(bleu))
        max_bleu = bleu_evaluator([ref], [ref])
        print("NM BLEU max: {}".format(max_bleu))
        min_bleu = bleu_evaluator([], [ref])
        print("NM BLEU min: {}".format(min_bleu))
        self.assertGreater(max_bleu, min_bleu)
        self.assertGreater(bleu, min_bleu)
        self.assertGreaterEqual(max_bleu, bleu)

    def test_beer(self):
        beer_evaluator = BeerWrapper("scripts/beer_2.0/beer")
        beer = beer_evaluator([hyp], [ref])
        print("BEER: {}".format(beer))
        max_beer = beer_evaluator([ref], [ref])
        print("BEER max: {}".format(max_beer))
        min_beer = beer_evaluator([], [ref])
        print("BEER min: {}".format(min_beer))
        self.assertGreater(max_beer, min_beer)
        self.assertGreater(beer, min_beer)
        self.assertGreaterEqual(max_beer, beer)


if __name__ == "__main__":
    unittest.main()
