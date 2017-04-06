import argparse
import wmt_client_python
from wmt_client_python.rest import ApiException
import numpy as np


def create_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Reduce lexical translations probabilities to the word"
                    "translation pairs with maximum probability")

    parser.add_argument(
        '--max_pairs', '-i', type=argparse.FileType('r'),
        metavar='PATH',
        help='Input file: word pairs with maximal probabilities')
    parser.add_argument(
        '--key_file', '-o', type=argparse.FileType('r'),
        metavar='PATH',
        help='File with API key')
    parser.add_argument(
        '--endpoint', '-e', type=argparse.FileType('r'),
        metavar='PATH',
        help='File with endpoint'
    )
    parser.add_argument(
        '--verbose', '-v', action='store_true',
        help='verbose mode.')

    return parser

def read_lex(input_file, encoding="utf-8", threshold=0.0):
    lex_translation_dict = dict()
    for line in input_file:
        source, target, logprob = line.strip().split()
        if float(logprob) > np.log(threshold):
            lex_translation_dict[source] = target
    return lex_translation_dict


if __name__ == '__main__':

    parser = create_parser()
    args = parser.parse_args()

    lex_translation = read_lex(args.max_pairs)

    wmt_client_python.configuration.api_key[
        'x-api-key'] = args.key_file.readline().strip()
    wmt_client_python.configuration.host = args.endpoint.readline().strip()
    api_instance = wmt_client_python.SharedTaskApi()

    api_instance.reset_dataset()

    training = True
    seen_instances = 0
    sum_rewards = 0

    while training:
        # request the next source sentence
        wmt_sentence = None
        sentence_id = None
        while wmt_sentence is None:
            try:
                api_response = api_instance.get_sentence()
                wmt_sentence = api_response.source
                sentence_id = api_response.id
            except ApiException as e:
                print("Exception when calling get_sentence {}".format(e))
                if e.status == 404:
                    print("Training ended!")
                    training = False
                    break

        if not training:
            break

        seen_instances += 1

        source_tokens = wmt_sentence.split(" ")
        translated = " ".join([lex_translation.get(token, token) for token in source_tokens])
        t = wmt_client_python.Translation(id=sentence_id, translation=translated)
        r = None
        while r is None:
            try:
                translation_response = api_instance.send_translation(t)
                r = translation_response.score
            except ApiException as e:
                print("Exception when calling send_translation:"
                          " {}\n".format(e))

        sum_rewards += r

        if seen_instances % 10 == 0:
            print("input sentence: {}".format(wmt_sentence))
            print("translation: {}".format(translated))
            print("reward: {}".format(r))
            print("cum reward: {}".format(sum_rewards))
            print("avg reward: {}".format(sum_rewards/seen_instances))







