#!/usr/bin/env python

"""This is a training script for sequence to sequence learning.

"""

# tests: lint, mypy

import sys
import os
from shutil import copyfile

import tensorflow as tf

from neuralmonkey.checking import check_dataset_and_coders
from neuralmonkey.utils import print_header, log, set_log_file
from neuralmonkey.config.configuration import Configuration
from neuralmonkey.config.config_generator import save_configuration
from neuralmonkey.learning_utils import training_loop, initialize_tf
from neuralmonkey.dataset import Dataset

def create_config(config_file):
    config = Configuration()
    config.add_argument('name', str)
    config.add_argument('random_seed', int, required=False)
    config.add_argument('output', basestring)
    config.add_argument('epochs', int, cond=lambda x: x >= 0)
    config.add_argument('trainer')
    config.add_argument('encoders', list)
    config.add_argument('decoder')
    config.add_argument('batch_size', int, cond=lambda x: x > 0)
    config.add_argument('train_dataset', Dataset)
    config.add_argument('val_dataset', Dataset)
    config.add_argument('postprocess')
    config.add_argument('evaluation', cond=list)
    config.add_argument('runner')
    config.add_argument('test_datasets', list, required=False, default=[])
    config.add_argument('initial_variables', str, required=False, default=[])
    config.add_argument('validation_period', int, required=False, default=500)
    config.add_argument('logging_period', int, required=False, default=20)
    config.add_argument('threads', int, required=False, default=4)
    config.add_argument('minimize', bool, required=False, default=False)
    config.add_argument('save_n_best', int, required=False, default=1)
    config.add_argument('overwrite_output_dir', bool, required=False,
                        default=False)

    return config.load_file(config_file)

def main():

    args = create_config(sys.argv[1])

    print ""

    #pylint: disable=no-member,broad-except
    if args.random_seed is not None:
        tf.set_random_seed(args.random_seed)

    if os.path.isdir(args.output):
        if args.overwrite_output_dir:
            # we do not want to delete the directory contents
            log("Directory '{}' exists, overwriting enabled, proceeding."
                .format(args.output))
        else:
            log("Directory '{}' exists, overwriting disabled."
                .format(args.output), color='red')
            exit(1)

    try:
        check_dataset_and_coders(args.train_dataset,
                                 args.encoders + [args.decoder])
        check_dataset_and_coders(args.val_dataset,
                                 args.encoders + [args.decoder])
        for test in args.test_datasets:
            check_dataset_and_coders(test, args.encoders)
    except Exception as exc:
        log(exc.message, color='red')
        exit(1)

    if not os.path.isdir(args.output):
        try:
            os.mkdir(args.output)
        except Exception as exc:
            log("Failed to create experiment directory: {}. Exception: {}"
                    .format(args.output, exc), color='red')
            exit(1)

    log_file = "{}/experiment.log".format(args.output)
    ini_file = "{}/experiment.ini".format(args.output)
    git_commit_file = "{}/git_commit".format(args.output)
    git_diff_file = "{}/git_diff".format(args.output)
    variables_file_prefix = "{}/variables.data".format(args.output)

    cont_index = 0

    while (os.path.exists(log_file)
           or os.path.exists(ini_file)
           or os.path.exists(git_commit_file)
           or os.path.exists(git_diff_file)
           or os.path.exists(variables_file_prefix)
           or os.path.exists("{}.0".format(variables_file_prefix))):
        cont_index += 1

        log_file = "{}/experiment.log.cont-{}".format(args.output, cont_index)
        ini_file = "{}/experiment.ini.cont-{}".format(args.output, cont_index)
        git_commit_file = "{}/git_commit.cont-{}".format(
            args.output, cont_index)
        git_diff_file = "{}/git_diff.cont-{}".format(args.output, cont_index)
        variables_file_prefix = "{}/variables.data.cont-{}".format(
            args.output, cont_index)

    copyfile(sys.argv[1], ini_file)
    set_log_file(log_file)
    print_header(args.name)

    os.system("git log -1 --format=%H > {}".format(git_commit_file))
    os.system("git --no-pager diff --color=always > {}".format(git_diff_file))

    link_best_vars = "{}.best".format(variables_file_prefix)

    run_configuration = {
        'encoders': args.encoders,
        'decoder': args.decoder,
        'runner': args.runner,
        'evaluation': args.evaluation,
        'initial_variables': link_best_vars
    }
    save_configuration(run_configuration, args.output)

    sess, saver = initialize_tf(args.initial_variables, args.threads)
    training_loop(sess, saver, args.epochs, args.trainer,
                  args.encoders + [args.decoder], args.decoder,
                  args.batch_size, args.train_dataset, args.val_dataset,
                  args.output, args.evaluation, args.runner,
                  test_datasets=args.test_datasets,
                  save_n_best_vars=args.save_n_best,
                  link_best_vars=link_best_vars,
                  vars_prefix=variables_file_prefix,
                  logging_period=args.logging_period,
                  validation_period=args.validation_period,
                  postprocess=args.postprocess,
                  minimize_metric=args.minimize)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print "Usage: train.py <ini_file>"
        exit(1)
    else:
        main()