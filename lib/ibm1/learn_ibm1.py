from nltk.translate import AlignedSent, IBMModel1
import argparse
import operator


def create_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="learn lexical translations with IBM1 (NLTK)")

    parser.add_argument(
        '--source', '-s', type=argparse.FileType('r'),
        metavar='PATH',
        help='Training source text.')
    parser.add_argument(
        '--target', '-t', type=argparse.FileType('r'),
        metavar='PATH',
        help='Training target text.')
    parser.add_argument(
        '--output', '-o', type=argparse.FileType('w'),
        metavar='PATH',
        help='Location for output')
    parser.add_argument(
        '--iterations', '-i', type=int, default=100,
        help="Run the model for this many iterations")
    parser.add_argument(
        '--verbose', '-v', action="store_true",
        help="verbose mode.")

    return parser


def bitext_from_files(source_file, target_file):
    bitext = []
    c = 0
    for source_line, target_line in zip(source_file, target_file):
        source_tokens = source_line.strip().split()
        target_tokens = target_line.strip().split()
        bitext.append(AlignedSent(source_tokens, target_tokens))
        c += 1
    return bitext

if __name__ == '__main__':

    parser = create_parser()
    args = parser.parse_args()

    bitext = bitext_from_files(args.source, args.target)
    if args.verbose:
        print("Read {} sentences to bitext".format(len(bitext)))

    if args.verbose:
        print("Training IBM 1 model for {} iterations.".format(args.iterations))

    ibm1 = IBMModel1(bitext, iterations=args.iterations)
    if args.verbose:
        print("Build IBM 1 model.")

    output_file = args.output
    print(output_file)
    for word1 in ibm1.translation_table.keys():
        argmax = ""
        max = 0
        for word2, prob in ibm1.translation_table[word1].items():
            if prob >= max:
                max = prob
                argmax = word2
        output_file.write("{} {} {}\n".format(word1, argmax, max))
    if args.verbose:
        print("Dumped IBM 1 model translation prob maxes to {}".format(output_file))

    #test_word = "Junge"
    #translation_probs = ibm1.translation_table[test_word]
    #print(translation_probs)
    #sorted_probs = sorted(translation_probs.items(), key=operator.itemgetter(1), reverse=True)
    #print(sorted_probs[:10])