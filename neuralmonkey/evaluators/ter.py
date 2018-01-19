import pyter


# pylint: disable=too-few-public-methods
class TEREvaluator(object):
    """Compute TER using the pyter library."""

    def __init__(self, name: str = "TER", negate: bool = False) -> None:
        self.name = name
        self.negate = negate

    def __call__(self, decoded, references) -> float:
        ter_sum = 0.
        count = 0
        for hyp, ref in zip(decoded, references):
            count += 1
            if ref and hyp:
                ter_sum += pyter.ter(hyp, ref)
            elif not ref and not hyp:
                ter_sum += 0.
            else:
                ter_sum += 1.
        if not self.negate:
            return ter_sum / count
        else:
            return (1-ter_sum) / count


TER = TEREvaluator()
