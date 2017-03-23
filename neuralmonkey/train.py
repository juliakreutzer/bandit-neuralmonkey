"""
This is a training script for sequence to sequence learning.
"""
# tests: lint, mypy

import sys
import random
import os
from shutil import copyfile
import numpy as np
import tensorflow as tf

from neuralmonkey.checking import CheckingException, check_dataset_and_coders
from neuralmonkey.logging import Logging, log
from neuralmonkey.config.configuration import Configuration
from neuralmonkey.learning_utils import training_loop, bandit_training_loop, bandit_training_loop_wmt
from neuralmonkey.dataset import Dataset
from neuralmonkey.tf_manager import TensorFlowManager
from neuralmonkey.trainers.generic_bandit_trainer import GenericBanditTrainer


def create_config() -> Configuration:
    config = Configuration()

    # training loop arguments
    config.add_argument('tf_manager', TensorFlowManager)
    config.add_argument('epochs', int, cond=lambda x: x >= 0, required=False,
                        default=20)
    config.add_argument('trainer')
    config.add_argument('batch_size', int, cond=lambda x: x > 0, required=False,
                        default=1)
    config.add_argument('train_dataset', Dataset, required=False)
    config.add_argument('val_dataset', Dataset, required=False)
    config.add_argument('output', str)
    config.add_argument('evaluation', list)
    config.add_argument('runners', list, required=False, default=[])
    config.add_argument('test_datasets', list, required=False, default=[])
    config.add_argument('logging_period', int, required=False, default=20)
    config.add_argument('validation_period', int, required=False, default=500)
    config.add_argument('runners_batch_size', int,
                        required=False, default=None)
    config.add_argument('minimize', bool, required=False, default=False)
    config.add_argument('postprocess')
    config.add_argument('copypostprocess')
    config.add_argument('name', str)
    config.add_argument('random_seed', int, required=False)
    config.add_argument('initial_variables', str, required=False, default=[])
    config.add_argument('overwrite_output_dir', bool, required=False,
                        default=False)
    config.add_argument('batch_reward', bool, required=False, default=False)
    config.add_argument('wmt', bool, required=False, default=False)
    config.add_argument('save_every_n', int, required=False, default=100)

    return config


# pylint: disable=too-many-statements
def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: train.py <ini_file>")
        exit(1)

    # define valid parameters and defaults
    cfg = create_config()
    # load the params from the config file, getting also the simple arguments
    cfg.load_file(sys.argv[1])
    # various things like randseed or summarywriter should be set up here
    # so that graph building can be recorded
    # build all the objects specified in the config

    if cfg.args.random_seed is None:
        cfg.args.random_seed = 2574600
    random.seed(cfg.args.random_seed)
    np.random.seed(cfg.args.random_seed)
    tf.set_random_seed(cfg.args.random_seed)

    cfg.build_model()

    if len(cfg.args.initial_variables) > 0:
        cfg.model.tf_manager.restore(cfg.args.initial_variables)
        log("Restoring model from {}".format(cfg.args.initial_variables))
    else:
        log("Building model from scratch")

    # pylint: disable=no-member
    if (os.path.isdir(cfg.model.output) and
            os.path.exists(os.path.join(cfg.model.output, "experiment.ini"))):
        if cfg.model.overwrite_output_dir:
            # we do not want to delete the directory contents
            log("Directory with experiment.ini '{}' exists, "
                "overwriting enabled, proceeding."
                .format(cfg.model.output))
        else:
            log("Directory with experiment.ini '{}' exists, "
                "overwriting disabled."
                .format(cfg.model.output), color='red')
            exit(1)

    if not cfg.model.wmt:
        # don't check for datasets for wmt training
        try:
            check_dataset_and_coders(cfg.model.train_dataset,
                                     cfg.model.runners)
            check_dataset_and_coders(cfg.model.val_dataset,
                                     cfg.model.runners)
        except CheckingException as exc:
            log(str(exc), color='red')
            exit(1)

    # pylint: disable=broad-except
    if not os.path.isdir(cfg.model.output):
        try:
            os.mkdir(cfg.model.output)
        except Exception as exc:
            log("Failed to create experiment directory: {}. Exception: {}"
                .format(cfg.model.output, exc), color='red')
            exit(1)

    log_file = "{}/experiment.log".format(cfg.model.output)
    gradients_file = "{}/gradients.log".format(cfg.model.output)
    updates_file = "{}/updates.log".format(cfg.model.output)
    rewards_file = "{}/rewards.log".format(cfg.model.output)
    ini_file = "{}/experiment.ini".format(cfg.model.output)
    git_commit_file = "{}/git_commit".format(cfg.model.output)
    git_diff_file = "{}/git_diff".format(cfg.model.output)
    variables_file_prefix = "{}/variables.data".format(cfg.model.output)

    cont_index = 0

    while (os.path.exists(log_file)
           or os.path.exists(ini_file)
           or os.path.exists(git_commit_file)
           or os.path.exists(git_diff_file)
           or os.path.exists(variables_file_prefix)
           or os.path.exists("{}.0".format(variables_file_prefix))):
        cont_index += 1

        log_file = "{}/experiment.log.cont-{}".format(
            cfg.model.output, cont_index)
        gradients_file = "{}/gradients.log.cont-{}".format(
            cfg.model.output, cont_index)
        updates_file = "{}/updates.log.cont-{}".format(
            cfg.model.output, cont_index)
        rewards_file = "{}/rewards.log.cont-{}".format(
            cfg.model.output, cont_index)
        ini_file = "{}/experiment.ini.cont-{}".format(
            cfg.model.output, cont_index)
        git_commit_file = "{}/git_commit.cont-{}".format(
            cfg.model.output, cont_index)
        git_diff_file = "{}/git_diff.cont-{}".format(
            cfg.model.output, cont_index)
        variables_file_prefix = "{}/variables.data.cont-{}".format(
            cfg.model.output, cont_index)

    copyfile(sys.argv[1], ini_file)
    Logging.set_log_file(log_file)
    Logging.print_header(cfg.model.name)

    # this points inside the neuralmonkey/ dir inside the repo, but
    # it does not matter for git.
    repodir = os.path.dirname(os.path.realpath(__file__))

    os.system("cd {}/..; git log -1 --format=%H > {}"
              .format(repodir, git_commit_file))

    os.system("cd {}/..; git --no-pager diff --color=always > {}"
              .format(repodir, git_diff_file))

    link_best_vars = "{}.best".format(variables_file_prefix)

    # runners_batch_size must be set to avoid problems on GPU
    if cfg.model.runners_batch_size is None:
        cfg.model.runners_batch_size = cfg.model.batch_size

    if isinstance(cfg.model.trainer, GenericBanditTrainer) and cfg.model.wmt:
            log("Online Learning for WMT")
            bandit_training_loop_wmt(tf_manager=cfg.model.tf_manager,
                      trainer=cfg.model.trainer,
                      evaluators=cfg.model.evaluation,
                      vars_prefix=variables_file_prefix,
                      logging_period=cfg.model.logging_period,
                      postprocess=cfg.model.postprocess,
                      copypostprocess=cfg.model.copypostprocess,
                      save_every_n=cfg.model.save_every_n)
    else:

        if isinstance(cfg.model.trainer, GenericBanditTrainer):
            loop_function = bandit_training_loop
            log("Bandit Training.")

        else:
            loop_function = training_loop
            log("Full Information Training.")

        try:
            # only used for bandit learning
            store_gradients = cfg.model.trainer.store_gradients
        except:
            store_gradients = False

        loop_function(tf_manager=cfg.model.tf_manager,
                      epochs=cfg.model.epochs,
                      trainer=cfg.model.trainer,
                      batch_size=cfg.model.batch_size,
                      train_dataset=cfg.model.train_dataset,
                      val_dataset=cfg.model.val_dataset,
                      log_directory=cfg.model.output,
                      valuelog_dirs=[gradients_file, updates_file, rewards_file],
                      evaluators=cfg.model.evaluation,
                      runners=cfg.model.runners,
                      test_datasets=cfg.model.test_datasets,
                      link_best_vars=link_best_vars,
                      vars_prefix=variables_file_prefix,
                      logging_period=cfg.model.logging_period,
                      validation_period=cfg.model.validation_period,
                      postprocess=cfg.model.postprocess,
                      copypostprocess=cfg.model.copypostprocess,
                      runners_batch_size=cfg.model.runners_batch_size,
                      minimize_metric=cfg.model.minimize,
                      store_gradients=store_gradients,
                      batch_reward=cfg.model.batch_reward)
