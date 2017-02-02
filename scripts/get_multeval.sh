#!/usr/bin/env bash
# Get multeval scripts
# See: https://github.com/jhclark/multeval

wget http://www.cs.cmu.edu/~jhclark/downloads/multeval-0.5.1.tgz
tar -xvzf multeval-0.5.1.tgz

# run an example to download METEOR

cp multeval-0.5.1/constants ../


multeval-0.5.1/multeval.sh eval --refs multeval-0.5.1/example/refs.test2010.lc.tok.en.* \
                   --hyps-baseline multeval-0.5.1/example/hyps.lc.tok.en.baseline.opt* \
                   --meteor.language en

