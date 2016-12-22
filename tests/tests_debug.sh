#!/bin/bash

set -ex

bin/neuralmonkey-train tests/small.ini
bin/neuralmonkey-run tests/small.ini tests/test_data.ini

rm -rf tests/tmp-test-output
