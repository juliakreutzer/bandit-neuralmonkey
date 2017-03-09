import re
from typing import List


class UNKCopyPostProcessor(object):
    """
    Post-processes target and source sentence pairs by replacing target UNKs
    with indicated source words
    """
    def __init__(self, **kwargs):
        self.separator = kwargs.get("separator", "-")
        self.unk_symbol = kwargs.get("unk_symbol", "UNK")

        esc = re.escape(self.separator)
        self.pattern = \
            re.compile(self.unk_symbol+esc+"(?P<source_index>[0-9]+)")

    def __call__(self, source_sentences: List[List[str]],
                 decoded_sentences: List[List[str]]) -> List[List[str]]:
        return [self.decode(s, t) for s, t in
                zip(source_sentences, decoded_sentences)]

    def decode(self, source_sentence: List[str], target_sentence: List[str])\
                -> List[str]:
        """
        Replace "UNK-i" in target output with i-th word of source
        :param source_sentence:
        :param target_sentence:
        :return:
        """
        for i, token in enumerate(target_sentence):
            match = self.pattern.match(token)
            if match is not None:
                replace_index = int(match.group("source_index"))
                if replace_index >= len(source_sentence):
                    # aligned to padding
                    target_sentence[i] = ""
                else:
                    target_sentence[i] = source_sentence[replace_index]

        return target_sentence

