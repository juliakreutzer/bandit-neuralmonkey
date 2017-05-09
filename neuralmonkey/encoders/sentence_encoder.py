# tests: lint, mypy

from typing import Optional, Dict, Any, Tuple

import tensorflow as tf

from neuralmonkey.encoders.attentive import Attentive
from neuralmonkey.model.model_part import ModelPart, FeedDict
from neuralmonkey.logging import log
from neuralmonkey.nn.noisy_gru_cell import NoisyGRUCell as NoisyActivatedGRUCell
from neuralmonkey.noise_utils import NoisyOrthoGRUCell, noisy_dynamic_rnn
from neuralmonkey.nn.ortho_gru_cell import OrthoGRUCell
from neuralmonkey.dataset import Dataset
from neuralmonkey.vocabulary import Vocabulary
from neuralmonkey.nn.utils import dropout


# pylint: disable=invalid-name
AttType = Any  # Type[] or union of types do not work here
RNNCellTuple = Tuple[tf.nn.rnn_cell.RNNCell, tf.nn.rnn_cell.RNNCell]
# pylint: enable=invalid-name


# pylint: disable=too-many-instance-attributes
class SentenceEncoder(ModelPart, Attentive):
    """A class that manages parts of the computation graph that are
    used for encoding of input sentences. It uses a bidirectional RNN.

    This version of the encoder does not support factors. Should you
    want to use them, use FactoredEncoder instead.
    """

    # pylint: disable=too-many-arguments,too-many-locals
    def __init__(self,
                 name: str,
                 vocabulary: Vocabulary,
                 data_id: str,
                 max_input_len: int,
                 embedding_size: int,
                 rnn_size: int,
                 rnn_cell: str="GRU",
                 dropout_keep_prob: float=1.0,
                 attention_type: Optional[AttType]=None,
                 attention_fertility: int=3,
                 use_noisy_activations: bool=False,
                 parent_encoder: Optional["SentenceEncoder"]=None,
                 save_checkpoint: Optional[str]=None,
                 load_checkpoint: Optional[str]=None,
                 train_mode: bool=False,
                 delta: float=1.0) -> None:
        """Createes a new instance of the sentence encoder

        Arguments:
            vocabulary: Input vocabulary
            data_id: Identifier of the data series fed to this encoder
            name: An unique identifier for this encoder
            max_input_len: Maximum length of an encoded sequence
            embedding_size: The size of the embedding vector assigned
                to each word
            rnn_size: The size of the encoder's hidden state. Note
                that the actual encoder output state size will be
                twice as long because it is the result of
                concatenation of forward and backward hidden states.
            train_mode: during training noise is added, but not during testing

        Keyword arguments:
            dropout_keep_prob: The dropout keep probability
                (default 1.0)
            attention_type: The class that is used for creating
                attention mechanism (default None)
            attention_fertility: Fertility parameter used with
                CoverageAttention (default 3).
        """
        ModelPart.__init__(self, name, save_checkpoint, load_checkpoint)
        Attentive.__init__(
            self, attention_type, attention_fertility=attention_fertility)

        self.vocabulary = vocabulary
        self.data_id = data_id

        self.max_input_len = max_input_len
        self.embedding_size = embedding_size
        self.rnn_size = rnn_size
        self.rnn_cell = rnn_cell
        self.dropout_keep_p = dropout_keep_prob
        self.use_noisy_activations = use_noisy_activations
        self.parent_encoder = parent_encoder

        log("Initializing sentence encoder, name: '{}'"
            .format(self.name))

        with tf.variable_scope(self.name):
            self._create_input_placeholders()
            with tf.variable_scope('input_projection'):
                self._create_embedding_matrix()
                embedded_inputs = self._embed(self.inputs)  # type: tf.Tensor

            fw_cell, bw_cell = self.rnn_cells()  # type: RNNCellTuple


            # from tf.nn.bidirectional_dynamic_rnn
            time_major = False  # default

            with tf.variable_scope("BiRNN"):
                # Forward direction
                with tf.variable_scope("FW") as fw_scope:
                    output_fw, output_state_fw, gradient_fw = noisy_dynamic_rnn(
                        cell=fw_cell, inputs=embedded_inputs,
                        sequence_length=self.sentence_lengths,
                        initial_state=None, dtype=tf.float32,
                        parallel_iterations=None,
                        swap_memory=None,
                        time_major=time_major, scope=fw_scope,
                        train_mode=train_mode, direction="fw", delta=delta)

                # Backward direction
                if not time_major:
                    time_dim = 1
                    batch_dim = 0
                else:
                    time_dim = 0
                    batch_dim = 1

                with tf.variable_scope("BW") as bw_scope:
                    inputs_reverse = tf.reverse_sequence(
                        input=embedded_inputs, seq_lengths=self.sentence_lengths,
                        seq_dim=time_dim, batch_dim=batch_dim)
                    tmp, output_state_bw, gradient_bw = noisy_dynamic_rnn(
                        cell=bw_cell, inputs=inputs_reverse,
                        sequence_length=self.sentence_lengths,
                        initial_state=None, dtype=tf.float32,
                        parallel_iterations=None,
                        swap_memory=None,
                        time_major=time_major, scope=bw_scope,
                        train_mode=train_mode, direction="bw", delta=delta)

            output_bw = tf.reverse_sequence(
                input=tmp, seq_lengths=self.sentence_lengths,
                seq_dim=time_dim, batch_dim=batch_dim)

            self.outputs_bidi_tup = (output_fw, output_bw)
            encoded_tup = (output_state_fw, output_state_bw)

            self.hidden_states = tf.concat(2, self.outputs_bidi_tup)

            with tf.variable_scope('attention_tensor'):
                self.__attention_tensor = dropout(
                    self.hidden_states, self.dropout_keep_p, self.train_mode)

            self.encoded = tf.concat(1, encoded_tup)
            self.gradients = gradient_fw + gradient_bw  # list concat

        log("Sentence encoder initialized")

    @property
    def _attention_tensor(self):
        return self.__attention_tensor

    @property
    def _attention_mask(self):
        return self._input_mask

    @property
    def vocabulary_size(self):
        return len(self.vocabulary)

    def _get_placeholders(self):
        """
        Get all the placeholders of the encoder
        :return:
        """
        placeholders = [self.train_mode, self.inputs, self._input_mask]
        return placeholders

    def _create_input_placeholders(self):
        """Creates input placeholder nodes in the computation graph"""
        self.train_mode = tf.placeholder(tf.bool, shape=[],
                                         name="mode_placeholder")

        self.inputs = tf.placeholder(tf.int32,
                                     shape=[None, self.max_input_len],
                                     name="encoder_input")

        self._input_mask = tf.placeholder(
            tf.float32, shape=[None, self.max_input_len],
            name="encoder_padding")

        self.sentence_lengths = tf.to_int32(
            tf.reduce_sum(self._input_mask, 1))

    def _create_embedding_matrix(self):
        """Create variables and operations for embedding the input words.

        If parent encoder is specified, we reuse its embedding matrix
        """
        # NOTE the note from the decoder's embedding matrix function applies
        # here also
        if self.parent_encoder is not None:
            self.embedding_matrix = self.parent_encoder.embedding_matrix
        else:
            self.embedding_matrix = tf.get_variable(
                "word_embeddings", [self.vocabulary_size, self.embedding_size],
                initializer=tf.random_normal_initializer(stddev=0.01))

    def _embed(self, inputs: tf.Tensor) -> tf.Tensor:
        """Embed the input using the embedding matrix and apply dropout

        Arguments:
            inputs: The Tensor to be embedded and dropped out.
        """
        embedded = tf.nn.embedding_lookup(self.embedding_matrix, inputs)
        return dropout(embedded, self.dropout_keep_p, self.train_mode)

    def rnn_cells(self) -> RNNCellTuple:
        """Return the graph template to for creating RNN memory cells"""

        if self.parent_encoder is not None:
            return self.parent_encoder.rnn_cells()

        if self.use_noisy_activations:
            return(NoisyActivatedGRUCell(self.rnn_size, self.train_mode),
                   NoisyActivatedGRUCell(self.rnn_size, self.train_mode))

        if self.rnn_cell == "NoisyOrthoGRU":
            return (NoisyOrthoGRUCell(self.rnn_size), NoisyOrthoGRUCell(self.rnn_size))

        return (OrthoGRUCell(self.rnn_size),
                OrthoGRUCell(self.rnn_size))

    def feed_dict(self, dataset: Dataset, train: bool=False) -> FeedDict:
        """Populate the feed dictionary with the encoder inputs.

        Encoder input placeholders:
            ``encoder_input``: Stores indices to the vocabulary,
                shape (batch, time)
            ``encoder_padding``: Stores the padding (ones and zeros,
                indicating valid words and positions after the end
                of sentence, shape (batch, time)
            ``train_mode``: Boolean scalar specifying the mode (train
                vs runtime)

        Arguments:
            dataset: The dataset to use
            train: Boolean flag telling whether it is training time
        """
        # pylint: disable=invalid-name
        fd = {}  # type: FeedDict
        fd[self.train_mode] = train
        sentences = dataset.get_series(self.data_id)

        vectors, paddings = self.vocabulary.sentences_to_tensor(
            list(sentences), self.max_input_len, train_mode=train)

        # as sentences_to_tensor returns lists of shape (time, batch),
        # we need to transpose
        fd[self.inputs] = list(zip(*vectors))
        fd[self._input_mask] = list(zip(*paddings))

        return fd
