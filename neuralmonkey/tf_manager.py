"""Interface between the data and TF session.

This module impelements the TensorFlow manager which encapsulates the graph
execution in existing sessions.

"""

# pylint: disable=unused-import
from typing import Any, List, Union, Dict, Set
# pylint: enable=unused-import

import tensorflow as tf
from neuralmonkey.logging import log
from neuralmonkey.dataset import Dataset
from tensorflow.python.ops import variables


from neuralmonkey.runners.base_runner import (ExecutionResult,
                                              BanditExecutionResult,
                                              Executable,
                                              BanditExecutable,
                                              reduce_execution_results)

# tests: pylint,mypy


class TensorFlowManager(object):
    """Inteface between computational graph, data and TF sessions.

    Attributes:
        sessions: List of active Tensorflow sessions.
    """

    def __init__(self, num_sessions, num_threads, save_n_best=1,
                 variable_files=None, gpu_allow_growth=True,
                 per_process_gpu_memory_fraction=1.0,
                 report_gpu_memory_consumption=False,
                 api_key_file=None, host=None):
        """Initialize a TensorflowManager.

        At this moment the graph must already exist. This method initializes
        required number of TensorFlow sessions and initializes them with
        provided variable files if they are provided.

        Args:
            num_sessions: Number of sessions to be initialized.
            num_threads: Number of threads sessions will run in.
            variable_files: List of variable files.
            gpu_allow_growth: TF to allocate incrementally, not all at once.
            per_process_gpu_memory_fraction: Limit TF memory use.
            report_gpu_memory_consumption: Report overall GPU memory at every
                logging
        """

        session_cfg = tf.ConfigProto()
        session_cfg.inter_op_parallelism_threads = num_threads
        session_cfg.intra_op_parallelism_threads = num_threads
        session_cfg.allow_soft_placement = True  # needed for multiple GPUs
        # pylint: disable=no-member
        session_cfg.gpu_options.allow_growth = gpu_allow_growth
        session_cfg.gpu_options.per_process_gpu_memory_fraction = \
            per_process_gpu_memory_fraction
        self.report_gpu_memory_consumption = report_gpu_memory_consumption

        self.saver_max_to_keep = save_n_best
        self.sessions = [tf.Session(config=session_cfg)
                         for _ in range(num_sessions)]
        init_op = tf.initialize_all_variables()
        for sess in self.sessions:
            sess.run(init_op)

        self.saver = tf.train.Saver(max_to_keep=self.saver_max_to_keep)

        if api_key_file is not None:
            self.api_key = self.api_key_from_file(api_key_file)
        else:
            self.api_key = None

        self.host = host

        if variable_files:
            if len(variable_files) != num_sessions:
                raise Exception(("The number of provided variable files ({}) "
                                 "is different than a number sessions ({})")
                                .format(len(variable_files), num_sessions))
            self.restore(variable_files)


    # pylint: disable=too-many-locals
    def execute(self,
                dataset: Dataset,
                execution_scripts,
                train=False,
                compute_losses=True,
                summaries=True,
                batch_size=None) -> List[ExecutionResult]:
        if batch_size is None:
            batch_size = len(dataset)
        batched_dataset = dataset.batch_dataset(batch_size)

        batch_results = [
            [] for _ in execution_scripts]  # type: List[List[ExecutionResult]]
        batch_no = 0
        for batch in batched_dataset:
            print(batch_no)
            executables = [s.get_executable(compute_losses=compute_losses,
                                            summaries=summaries)
                           for s in execution_scripts]
            while not all(ex.result is not None for ex in executables):
                all_feedables = set()   # type: Set[Any]
                # type: Dict[Executable, tf.Tensor]
                all_tensors_to_execute = {}
                additional_feed_dicts = []
                tensor_list_lengths = []  # type: List[int]

                for executable in executables:
                    if executable.result is None:
                        (feedables,
                         tensors_to_execute,
                         add_feed_dict) = executable.next_to_execute()
                        all_feedables = all_feedables.union(feedables)
                        all_tensors_to_execute[executable] = tensors_to_execute
                        additional_feed_dicts.append(add_feed_dict)
                        tensor_list_lengths.append(len(tensors_to_execute))
                    else:
                        tensor_list_lengths.append(0)

                feed_dict = _feed_dicts(batch, all_feedables, train=train)
                for fdict in additional_feed_dicts:
                    feed_dict.update(fdict)

                session_results = [sess.run(all_tensors_to_execute,
                                            feed_dict=feed_dict)
                                   for sess in self.sessions]

                for executable in executables:
                    if executable.result is None:
                        executable.collect_results(
                            [res[executable] for res in session_results])

            for script_list, executable in zip(batch_results, executables):
                script_list.append(executable.result)
            batch_no += 1

        collected_results = []  # type: List[ExecutionResult]
        for result_list in batch_results:
            collected_results.append(reduce_execution_results(result_list))

        return collected_results

    def init_bandits(self, execution_scripts, summaries=False):
        # prepare partial run
        # need all feeds and all fetches
        all_feeds = []
        [all_feeds.extend(s.get_executable(summaries=summaries,
                                        update=True).get_feeds())
                       for s in execution_scripts]
        [all_feeds.extend(s.get_executable(summaries=summaries, update=False)
                          .get_feeds()) for s in execution_scripts]
        all_fetches = []
        [all_fetches.extend(s.get_executable(summaries=summaries,
                                             update=True).get_fetches())
                     for s in execution_scripts]
        [all_fetches.extend(s.get_executable(summaries=summaries,
                                             update=False).get_fetches()) for s
             in execution_scripts]

        # execution scripts include sampling and updating
        self.handlers = [sess.partial_run_setup(all_fetches, all_feeds) for sess
                         in self.sessions]


    def execute_bandits(self,
                        dataset: Dataset,
                        execution_scripts,
                        epoch,
                        step,
                        train=False,
                        summaries=True,
                        batch_size=None,
                        rewards=None,
                        baseline=None,
                        update=False,
                        store_gradients=False) -> List[BanditExecutionResult]:
        if batch_size is None:
            batch_size = len(dataset)
        batched_dataset = dataset.batch_dataset(batch_size)

        batch_results = [
            [] for _ in execution_scripts]
        # type: List[List[BanditExecutionResult]]
        for batch in batched_dataset:
            executables = [s.get_executable(summaries=summaries,
                                            update=update, store_gradients=store_gradients)
                           for s in execution_scripts]
            while not all(ex.result is not None for ex in executables):
                all_feedables = set()  # type: Set[Any]
                # type: Dict[BanditExecutable, tf.Tensor]
                all_tensors_to_execute = {}
                additional_feed_dicts = []
                tensor_list_lengths = []  # type: List[int]

                for executable in executables:
                    if executable.result is None:
                        (feedables,
                         tensors_to_execute,
                         add_feed_dict) = executable.next_to_execute(
                            reward=rewards, baseline=baseline, epoch=epoch,
                            step=step)

                        all_feedables = all_feedables.union(feedables)

                        all_tensors_to_execute[executable] = tensors_to_execute
                        additional_feed_dicts.append(add_feed_dict)
                        tensor_list_lengths.append(len(tensors_to_execute))
                    else:
                        tensor_list_lengths.append(0)

                feed_dict = _feed_dicts(batch, all_feedables, train=False)  # TODO is False because of bandit training

                for fdict in additional_feed_dicts:
                    feed_dict.update(fdict)

                if update:
                    update_dict = {}
                    for fdict in additional_feed_dicts:
                        update_dict.update(fdict)
                    feed_dict = update_dict  # FIXME hack to only feed additional reward feed

                session_results = [sess.partial_run(
                    h, all_tensors_to_execute, feed_dict=feed_dict)
                                   for sess, h in zip(self.sessions,
                                                      self.handlers)]

                # fill executable.results with fetched values
                for executable in executables:
                    if executable.result is None:
                        executable.collect_results(
                            [res[executable] for res in session_results])

            for script_list, executable in zip(batch_results, executables):
                script_list.append(executable.result)

        collected_results = []  # type: List[BanditExecutionResult]
        for result_list in batch_results:
            collected_results.append(reduce_execution_results(result_list))
        return collected_results

    def save(self, variable_files: Union[str, List[str]], global_step=0) \
            -> None:
        if isinstance(variable_files, str) and len(self.sessions) == 1:
            if global_step == 0:
                self.saver.save(self.sessions[0], variable_files)
            else:
                self.saver.save(self.sessions[0], variable_files,
                                global_step=global_step)
            return

        if isinstance(variable_files, str):
            variable_files = ["{}.{}".format(
                variable_files, i) for i in range(len(self.sessions))]

        if len(variable_files) != len(self.sessions):
            raise Exception(
                "Provided {} files for restoring {} sessions.".format(
                    len(variable_files), len(self.sessions)))

        for sess, file_name in zip(self.sessions, variable_files):
            if global_step == 0:
                self.saver.save(sess, file_name)
            else:
                self.saver.save(sess, file_name, global_step=global_step)

    def restore(self, variable_files: Union[str, List[str]]) -> None:
        if isinstance(variable_files, str):
            variable_files = [variable_files]
        if len(variable_files) != len(self.sessions):
            raise Exception(
                "Provided {} files for restoring {} sessions.".format(
                    len(variable_files), len(self.sessions)))

        for sess, file_name in zip(self.sessions, variable_files):
            log("Loading variables from {}".format(file_name))
            self.saver.restore(sess, file_name)

    def initialize_model_parts(self, runners) -> None:
        """Initialize model parts variables from their checkpoints."""

        all_coders = set.union(*[rnr.all_coders for rnr in runners])
        for coder in all_coders:
            for session in self.sessions:
                coder.load(session)

    def api_key_from_file(self, file):
        f = open(file, "r").read().strip()
        return f

    def get_api_key(self):
        return self.api_key

    def get_host(self):
        return self.host


def _feed_dicts(dataset, coders, train=False):
    """
    This function ensures all encoder and decoder objects feed their the data
    they need from the dataset.
    """
    res = {}

    for coder in coders:
        res.update(coder.feed_dict(dataset, train=train))

    return res
