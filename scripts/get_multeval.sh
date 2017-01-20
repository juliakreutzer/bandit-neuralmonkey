#!/usr/bin/env bash
# Get multeval scripts
# See: https://github.com/jhclark/multeval

wget http://www.cs.cmu.edu/~jhclark/downloads/multeval-0.5.1.tgz
tar -xvzf multeval-0.5.1.tgz

# run an example to download METEOR

./multeval.sh eval --refs example/refs.test2010.lc.tok.en.* \
                   --hyps-baseline example/hyps.lc.tok.en.baseline.opt* \
                   --meteor.language en

cp multeval-0.5.1/constants ../