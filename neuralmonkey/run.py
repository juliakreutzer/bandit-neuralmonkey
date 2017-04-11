# tests: lint, mypy

import sys
import os

from neuralmonkey.logging import log, log_print
from neuralmonkey.config.configuration import Configuration
from neuralmonkey.learning_utils import (evaluation, run_on_dataset,
                                         print_final_evaluation)
from neuralmonkey.tf_manager import TensorFlowManager
import wmt_client_python
from wmt_client_python.rest import ApiException
from neuralmonkey.dataset import Dataset


CONFIG = Configuration()
CONFIG.add_argument('tf_manager', TensorFlowManager)
CONFIG.add_argument('output', str)
CONFIG.add_argument('postprocess')
CONFIG.add_argument('copypostprocess')
CONFIG.add_argument('evaluation', list)
CONFIG.add_argument('runners', list)
CONFIG.add_argument('preprocess', required=False, default=None)
CONFIG.add_argument('threads', int, required=False, default=4)
CONFIG.add_argument('runners_batch_size', int, required=False, default=None)
CONFIG.add_argument('wmt', bool, required=False, default=False)
CONFIG.add_argument('initial_seen_instances', int, required=False, default=0)
# ignore arguments which are just for training
CONFIG.ignore_argument('val_dataset')
CONFIG.ignore_argument('trainer')
CONFIG.ignore_argument('name')
CONFIG.ignore_argument('train_dataset')
CONFIG.ignore_argument('epochs')
CONFIG.ignore_argument('batch_size')
CONFIG.ignore_argument('test_datasets')
CONFIG.ignore_argument('initial_variables')
CONFIG.ignore_argument('validation_period')
CONFIG.ignore_argument('logging_period')
CONFIG.ignore_argument('minimize')
CONFIG.ignore_argument('random_seed')
CONFIG.ignore_argument('save_n_best')
CONFIG.ignore_argument('overwrite_output_dir')
CONFIG.ignore_argument('store_gradients')
CONFIG.ignore_argument('batch_reward')
CONFIG.ignore_argument('initial_baseline')
CONFIG.ignore_argument('initial_steps')


def default_variable_file(output_dir):
    variables_file = os.path.join(output_dir, "variables.data.best")
    cont_index = 1

    def continuation_file():
        return os.path.join(output_dir,
                            "variables.data.cont-{}.best".format(cont_index))
    while os.path.exists(continuation_file()):
        variables_file = continuation_file()
        cont_index += 1

    return variables_file


def initialize_for_running(output_dir, tf_manager, variable_files) -> None:
    """Restore either default variables of from configuration.

    Arguments:
       output_dir: Training output directory.
       tf_manager: TensorFlow manager.
       variable_files: Files with variables to be restored or None if the
           default variables should be used.
    """
    # pylint: disable=no-member
    log_print("")

    if variable_files is None:
        default_varfile = default_variable_file(output_dir)

        log("Default variable file '{}' will be used for loading variables."
            .format(default_varfile))

        variable_files = [default_varfile]

    for vfile in variable_files:
        if not os.path.exists(vfile):
            log("Variable file {} does not exist".format(vfile),
                color="red")
            exit(1)

    tf_manager.restore(variable_files)

    log_print("")


def main() -> None:

    CONFIG.load_file(sys.argv[1])
    CONFIG.build_model()

    if not CONFIG.model.wmt:

        # pylint: disable=no-member,broad-except
        if len(sys.argv) != 3:
            print("Usage: run.py <run_ini_file> <test_datasets>")
            exit(1)

        test_datasets = Configuration()
        test_datasets.add_argument('test_datasets')
        test_datasets.add_argument('variables')

        test_datasets.load_file(sys.argv[2])
        test_datasets.build_model()
        datasets_model = test_datasets.model
        initialize_for_running(CONFIG.model.output, CONFIG.model.tf_manager,
                               datasets_model.variables)

        print("")

        evaluators = [(e[0], e[0], e[1]) if len(e) == 2 else e
                      for e in CONFIG.model.evaluation]

        for dataset in datasets_model.test_datasets:
            execution_results, output_data = run_on_dataset(
                CONFIG.model.tf_manager, CONFIG.model.runners,
                dataset, CONFIG.model.postprocess, CONFIG.model.copypostprocess,
                write_out=True)
            # TODO what if there is no ground truth
            eval_result = evaluation(evaluators, dataset, CONFIG.model.runners,
                                     execution_results, output_data)
            if eval_result:
                print_final_evaluation(dataset.name, eval_result)

    else:

        # pylint: disable=no-member,broad-except
        if len(sys.argv) < 2:
            print("Usage: run.py <run_ini_file> [variables]")
            exit(1)

        variables = None  # default variables from output of ini file
        if len(sys.argv) == 3:
            variables = sys.argv[-1]

        initialize_for_running(CONFIG.model.output, CONFIG.model.tf_manager,
                               [variables])

        # WMT: only translating, not learning
        tf_manager = CONFIG.model.tf_manager
        preprocess = CONFIG.model.preprocess
        postprocess = CONFIG.model.postprocess
        copypostprocess = CONFIG.model.copypostprocess
        CONFIG.model.runners_batch_size = 1

        wmt_client_python.configuration.api_key[
            'x-api-key'] = tf_manager.get_api_key()
        wmt_client_python.configuration.host = tf_manager.get_host()
        api_instance = wmt_client_python.SharedTaskApi()

        api_instance.reset_dataset()

        log("Start translating")
        finished = False

        seen_instances = CONFIG.model.initial_seen_instances
        rewards = 0.0

        try:
            while not finished:
                # request the next source sentence
                wmt_sentence = None
                sentence_id = None
                while wmt_sentence is None:
                    try:
                        api_response = api_instance.get_sentence()
                        wmt_sentence = api_response.source
                        sentence_id = api_response.id
                    except ApiException as e:
                        print(
                            "Exception when calling get_sentence {}".format(e))
                        if e.status == 404:
                            print("Training ended!")
                            finished = True
                            break

                if finished:
                    break

                seen_instances += 1

                # received sentence as source series, preprocess (BPE)
                max_length = 50
                raw_text_tokenized = wmt_sentence.split(" ")[:max_length]
                input_dict = {"source": [raw_text_tokenized]}
                if preprocess is not None:
                    text_preprocessed = preprocess(raw_text_tokenized)
                    input_dict["source_bpe"] = [text_preprocessed]
                batch_dataset = Dataset("wmt_input", input_dict, {})

                # translate this sentence
                execution_results, output_data = run_on_dataset(
                    CONFIG.model.tf_manager, CONFIG.model.runners,
                    batch_dataset, CONFIG.model.postprocess,
                    CONFIG.model.copypostprocess,
                    write_out=False)

                sentence = output_data["target_greedy"]

                if copypostprocess is not None:
                    inputs = batch_dataset.get_series("source")
                    sentence = copypostprocess(inputs, sentence)

                # evaluate translation
                reward = None

                source_len = len(wmt_sentence)
                max_target_len = 50*source_len

                translation_str = " ".join(sentence[0])[:max_target_len]
                translation_id = sentence_id

                t = wmt_client_python.Translation(id=translation_id,
                                                  translation=translation_str)

                while reward is None:
                    try:
                        translation_response = api_instance.send_translation(t)
                        reward = translation_response.score
                    except ApiException as e:
                        log_print("Exception when calling send_translation:"
                                  " {}\n".format(e))

                rewards += reward

                if seen_instances % 10 == 10-1:
                    log_print("WMT incoming sentence {}: {}".format(
                        seen_instances, wmt_sentence))
                    if preprocess is not None:
                        log_print("preprocessed {}: {}".format(
                            seen_instances, preprocess(wmt_sentence.split(" "))))
                    log_print("Translation sent back {}: {}".format(
                        seen_instances, translation_str))
                    log_print("Score: {}".format(reward))
                    log_print("Avg score: {}".format(rewards/seen_instances))

        except KeyboardInterrupt:
            log("Training interrupted by user.")

        log("Translation finished.")




