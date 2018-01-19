import argparse
from neuralmonkey.evaluators.ter import TEREvaluator

def main():
    parser = argparse.ArgumentParser(description="Calculate corpus TER")
    parser.add_argument("reference", help="reference translation")
    parser.add_argument("hypothesis", help="translation hypothesis")
    args = parser.parse_args()

    TER = TEREvaluator()
    decoded = []
    references = []
    with open(args.reference, "r") as ref_file:
        for ref_line in ref_file:
            references.append(ref_line.strip().split())
    with open(args.hypothesis, "r") as hyp_file:
        for hyp_line in hyp_file:
            decoded.append(hyp_line.strip().split())
    assert len(decoded) == len(references)

    ter = TER(decoded=decoded, references=references)
    print("TER: {}".format(ter))


if __name__=="__main__":
    main()



