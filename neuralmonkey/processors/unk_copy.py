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
        self.pad_symbol = kwargs.get("pad_symbol", "<pad>")
        self.start_symbol = kwargs.get("start_symbol", "<s>")
        self.end_symbol = kwargs.get("end_symbol", "</s>")
        self.lexical_translations = self._read_lexical_translations(kwargs.get("lexical_translations", None))

        esc = re.escape(self.separator)
        self.pattern = \
            re.compile(self.unk_symbol+esc+"(?P<source_index>[0-9]+)")

    def __call__(self, source_sentences: List[List[str]],
                 decoded_sentences: List[List[str]]) -> List[List[str]]:
        return [self.decode(s, t) for s, t in
                zip(source_sentences, decoded_sentences)]

    def _read_lexical_translations(self, input_file, encoding: str="utf-8",
                                   threshold: float=0.0):
        """
        Read lexical translations from file
        :param input_file:
        :return:
        """
        if input_file is None:
            return dict()
        lex_translation_dict = dict()
        with open(input_file, "r", encoding=encoding) as opened_file:
            for line in opened_file:
                source, target, prob = line.strip().split()
                if float(prob) > threshold:
                    lex_translation_dict[source] = target
        return lex_translation_dict


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
                    replace_source_word = source_sentence[replace_index]
                    dict_lookup = self.lexical_translations.get(replace_source_word, replace_source_word)
                    target_sentence[i] = dict_lookup

        target_sentence = [t for t in target_sentence
                           if t != self.pad_symbol
                           and t != self.start_symbol
                           and t != self.end_symbol
                           and t != ""]

        return target_sentence

