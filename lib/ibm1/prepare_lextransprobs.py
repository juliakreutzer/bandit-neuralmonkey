import argparse

def create_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Reduce lexical translations probabilities to the word"
                    "translation pairs with maximum probability")

    parser.add_argument(
        '--input', '-i', type=argparse.FileType('r'),
        metavar='PATH',
        help='Input file: word pairs with probabilities (target - source - prob')
    parser.add_argument(
        '--output', '-o', type=argparse.FileType('w'),
        metavar='PATH',
        help='Output file: word pairs with maximum probabilities')
    parser.add_argument(
        '--verbose', '-v', action='store_true',
        help='verbose mode.')

    return parser


if __name__ == '__main__':

    parser = create_parser()
    args = parser.parse_args()

    current_source = ""
    current_max = -float('Inf')
    current_best_target = ""

    for line in args.input:
        parts = line.strip().split()
        if len(parts) != 4 or line.startswith("<eps>"):
            continue
        else:
            source = parts[0]
            target = parts[1]
            logprob = float(parts[2])  # the bigger the better
        if source == current_source:
            # old source, only remember target if higher than previous prob
            if logprob > current_max:
                current_best_target = target
                current_max = logprob
        else:
            # new source
            # first dump old pair
            if current_source != "":
                args.output.write("{} {} {}\n".format(current_source, current_best_target, current_max))
            # then start with new source
            current_source = source
            current_max = logprob
            current_best_target = target

    args.output.write(
            "{} {} {}\n".format(current_source, current_best_target,
                                current_max))
