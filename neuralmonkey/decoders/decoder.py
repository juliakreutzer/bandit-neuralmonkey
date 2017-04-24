# tests: lint

from typing import cast, Iterable, List, Callable, Optional, Union, Any, \
    Tuple, Dict
import math

import tensorflow as tf
import numpy as np

from neuralmonkey.dataset import Dataset
from neuralmonkey.vocabulary import Vocabulary, START_TOKEN
from neuralmonkey.model.model_part import ModelPart, FeedDict
from neuralmonkey.logging import log
from neuralmonkey.nn.utils import dropout
from neuralmonkey.encoders.attentive import Attentive
from neuralmonkey.nn.projection import linear
from neuralmonkey.decoders.encoder_projection import (
    linear_encoder_projection, concat_encoder_projection, empty_initial_state)
from neuralmonkey.decoders.output_projection import no_deep_output
from neuralmonkey.gradient_utils import init_grad


# pylint: disable=too-many-instance-attributes,too-few-public-methods
# Big decoder cannot be simpler. Not sure if refactoring
# it into smaller units would be helpful
class Decoder(ModelPart):
    """A class that manages parts of the computation graph that are
    used for the decoding.
    """

    # pylint: disable=too-many-arguments,too-many-locals
    # pylint: disable=too-many-arguments,too-many-locals
    def __init__(self,
                 encoders: List[Any],
                 vocabulary: Vocabulary,
                 data_id: str,
                 name: str,
                 max_output_len: int,
                 dropout_keep_prob: float,
                 rnn_size: Optional[int] = None,
                 embedding_size: Optional[int] = None,
                 output_projection: Optional[Callable[
                     [tf.Tensor, tf.Tensor,
                      List[tf.Tensor]], tf.Tensor]] = None,
                 encoder_projection: Optional[
                     Callable[[tf.Tensor, Optional[int], Optional[List[Any]]],
                              tf.Tensor]] = None,
                 use_attention: bool = False,
                 embeddings_encoder: Optional[Any] = None,
                 rnn_cell: str = 'GRU',
                 attention_on_input: bool = True,
                 save_checkpoint: Optional[str] = None,
                 load_checkpoint: Optional[str] = None,
                 sample_size=1,
                 temperature=1.0) -> None:
        """Create a refactored version of monster decoder.

        Arguments:
            encoders: Input encoders of the decoder
            vocabulary: Target vocabulary
            data_id: Target data series
            name: Name of the decoder. Should be unique accross all Neural
                Monkey objects
            max_output_len: Maximum length of an output sequence
            dropout_keep_prob: Probability of keeping a value during dropout

        Keyword arguments:
            rnn_size: Size of the decoder hidden state, if None set
                according to encoders.
            embedding_size: Size of embedding vectors for target words
            output_projection: How to generate distribution over vocabulary
                from decoder rnn_outputs
            encoder_projection: How to construct initial state from encoders
            use_attention: Flag whether to look at attention vectors of the
                encoders
            embeddings_encoder: Encoder to take embeddings from
            rnn_cell: RNN Cell used by the decoder (GRU or LSTM)
            attention_on_input: Flag whether attention from previous decoding
                step should be combined with the input in the next step.
        """
        ModelPart.__init__(self, name, save_checkpoint, load_checkpoint)
        log("Initializing decoder, name: '{}'".format(name))

        self.encoders = encoders
        self.vocabulary = vocabulary
        self.data_id = data_id
        self.max_output_len = max_output_len
        self.dropout_keep_prob = dropout_keep_prob
        self.embedding_size = embedding_size
        self.rnn_size = rnn_size
        self.output_projection = output_projection
        self.encoder_projection = encoder_projection
        self.use_attention = use_attention
        self.attention_on_input = attention_on_input
        self.embeddings_encoder = embeddings_encoder
        self._rnn_cell = rnn_cell
        self.temperature = temperature

        if self.embedding_size is None and self.embeddings_encoder is None:
            raise ValueError("You must specify either embedding size or the "
                             "encoder from which to reuse the embeddings ("
                             "e.g. set either 'embedding_size' or "
                             " 'embeddings_encoder' parameter)")

        if self.embeddings_encoder is not None:
            if self.embedding_size is not None:
                log("Warning: Overriding the embedding_size parameter with the"
                    " size of the reused embeddings from the encoder.",
                    color="red")

            self.embedding_size = (
                self.embeddings_encoder.embedding_matrix.get_shape()[1].value)

        if self.encoder_projection is None:
            if len(self.encoders) == 0:
                log("No encoder - language model only.")
                self.encoder_projection = empty_initial_state
            elif rnn_size is None:
                log("No rnn_size or encoder_projection: Using concatenation of"
                    " encoded states")
                self.encoder_projection = concat_encoder_projection
                self.rnn_size = sum(e.encoded.get_shape()[1].value
                                    for e in encoders)
            else:
                log("Using linear projection of encoders as the initial state")
                self.encoder_projection = linear_encoder_projection(
                    self.dropout_keep_prob)

        assert self.rnn_size is not None

        if self.output_projection is None:
            log("No output projection specified - using simple concatenation")
            self.output_projection = no_deep_output

        with tf.variable_scope(name):
            self._create_input_placeholders()
            self._create_training_placeholders()
            self._create_initial_state()
            self._create_embedding_matrix()

            with tf.name_scope("output_projection"):
                self.decoding_w = tf.get_variable(
                    "state_to_word_W", [self.rnn_size, len(self.vocabulary)],
                    initializer=tf.random_uniform_initializer(-0.5, 0.5))

                self.decoding_b = tf.get_variable(
                    "state_to_word_b", [len(self.vocabulary)],
                    initializer=tf.constant_initializer(
                        - math.log(len(self.vocabulary))))

            # last train input is unused in decoding functions
            # (just as target)
            embedded_train_inputs = self._embed_and_dropout(
                self.train_inputs[:-1])

            # attention has done dropout
            self.embedded_go_symbols = tf.nn.embedding_lookup(
                self.embedding_matrix, self.go_symbols)

            # fetch train attention objects
            self._train_attention_objects = {}
            # type: Dict[Attentive, tf.Tensor]
            if self.use_attention:
                with tf.name_scope("attention_object"):
                    self._train_attention_objects = {
                        e: e.create_attention_object()
                        for e in self.encoders
                        if isinstance(e, Attentive)}

            # decoding with training target word inputs
            _, _, _, _, self.train_logits, _ = self.attention_decoder(
                self.embedded_go_symbols,
                attention_on_input=self.attention_on_input,
                train_inputs=embedded_train_inputs,
                train_mode=True,
                temperature=self.temperature, store_logits=True)

            assert not tf.get_variable_scope().reuse
            tf.get_variable_scope().reuse_variables()

            # fetch runtime attention objects
            self._runtime_attention_objects = {}
            # type: Dict[Attentive, tf.Tensor]
            if self.use_attention:
                self._runtime_attention_objects = {
                    e: e.create_attention_object()
                    for e in self.encoders
                    if isinstance(e, Attentive)}

            # decoding during inference: without target word inputs
            self.runtime_rnn_outputs, self.runtime_rnn_states, decoded_ids, \
            self.decoded_logprobs, self.runtime_logits, _ = \
                self.attention_decoder(
                    self.embedded_go_symbols,
                    attention_on_input=attention_on_input,
                    train_mode=False,
                    sample_mode=0,
                    temperature=self.temperature, store_logits=True,
                    store_rnn_outputs=True, store_rnn_states=True)

            self.decoded = tf.expand_dims(tf.pack(decoded_ids), 2)

            # xent loss
            train_targets = tf.unpack(self.train_inputs)

            self.train_logprobs = [tf.nn.log_softmax(l)
                                   for l in self.train_logits]

            self.train_loss = tf.nn.seq2seq.sequence_loss(
                self.train_logits, train_targets,
                tf.unpack(self.train_padding))
            self.cost = self.train_loss

            self.runtime_loss = tf.nn.seq2seq.sequence_loss(
                self.runtime_logits, train_targets,
                tf.unpack(self.train_padding))

            # sampling
            self.sample_size = sample_size

            self.rewards = tf.placeholder(tf.float32, [None, self.sample_size],
                                          name="rewards")
            self.baseline = tf.placeholder(tf.float32, [],
                                           name="baseline")  # scalar

            self.epoch = tf.placeholder(tf.int32, [], name="epoch")
            self.step = tf.placeholder(tf.int32, [], name="step")

            log("Decoder initalized.")

    def _create_input_placeholders(self) -> None:
        """Creates input placeholder nodes in the computation graph"""
        self.train_mode = tf.placeholder(tf.bool, name="decoder_train_mode")

        self.go_symbols = tf.placeholder(tf.int32, shape=[1, None],
                                         name="decoder_go_symbols")

        self.batch_size = tf.shape(self.go_symbols)[1]

    def _create_training_placeholders(self) -> None:
        """Creates training placeholder nodes in the computation graph

        The training placeholder nodes are NOT fed during runtime.
        """
        self.train_inputs = tf.placeholder(
            tf.int32, [self.max_output_len, None],
            name="decoder_input_placeholder")

        self.train_padding = tf.placeholder(
            tf.float32, [self.max_output_len, None],
            name="decoder_padding_placeholder")

    def _create_initial_state(self) -> None:
        """Construct the part of the computation graph that computes
        the initial state of the decoder.
        """
        with tf.variable_scope("initial_state"):
            self.initial_state = dropout(
                self.encoder_projection(self.train_mode,
                                        self.rnn_size,
                                        self.encoders),
                self.dropout_keep_prob,
                self.train_mode)

            # pylint: disable=no-member
            # Pylint keeps complaining about initial shape being a tuple,
            # but it is a tensor!!!
            init_state_shape = self.initial_state.get_shape()
            # pylint: enable=no-member

            # Broadcast the initial state to the whole batch if needed
            if len(init_state_shape) == 1:
                assert init_state_shape[0].value == self.rnn_size
                tiles = tf.tile(self.initial_state,
                                tf.expand_dims(self.batch_size, 0))
                self.initial_state = tf.reshape(tiles, [-1, self.rnn_size])

    def _create_embedding_matrix(self) -> None:
        """Create variables and operations for embedding of input words

        If we are reusing word embeddings, this function takes the embedding
        matrix from the first encoder
        """
        if self.embeddings_encoder is None:
            # TODO better initialization
            self.embedding_matrix = tf.get_variable(
                "word_embeddings", [len(self.vocabulary), self.embedding_size],
                initializer=tf.random_uniform_initializer(-0.5, 0.5))
        else:
            self.embedding_matrix = self.embeddings_encoder.embedding_matrix

    def _embed_and_dropout(self, inputs: tf.Tensor) -> tf.Tensor:
        """Embed the input using the embedding matrix and apply dropout

        Arguments:
            inputs: The Tensor to be embedded and dropped out.
        """
        with tf.variable_scope("embed_inputs"):
            embedded = tf.nn.embedding_lookup(
                self.embedding_matrix, inputs)
            return dropout(embedded,
                           self.dropout_keep_prob,
                           self.train_mode)

    def _logit_function(self, state: tf.Tensor) -> tf.Tensor:
        state = dropout(state, self.dropout_keep_prob, self.train_mode)
        return tf.matmul(state, self.decoding_w) + self.decoding_b

    def _get_rnn_cell(self) -> tf.nn.rnn_cell.RNNCell:
        if self._rnn_cell == 'GRU':
            return tf.nn.rnn_cell.GRUCell(self.rnn_size)
        elif self._rnn_cell == 'LSTM':
            return tf.nn.rnn_cell.LSTMCell(self.rnn_size)
        else:
            raise ValueError("Unknown RNN cell: {}".format(self._rnn_cell))

    def get_attention_object(self, encoder, train_mode: bool):
        if train_mode:
            return self._train_attention_objects.get(encoder)
        else:
            return self._runtime_attention_objects.get(encoder)

    def sample_from_runtime_logits(self, neg=False):
        """Sample from the runtime hidden states"""
        sample_size = 1  # for now
        voc_size = len(self.vocabulary)

        sample_ids = []
        sample_logprob = 0.0

        model_logprob = self.runtime_logits

        # sampling from negative weights of last layer
        ix = tf.constant(-1)
        if neg:
            # negating logits only for one word, chosen randomly
            ix = tf.random_uniform((1,), 0, len(self.runtime_logits),
                                   tf.int32)
            ixtemp = tf.one_hot(ix, len(self.runtime_logits), on_value=-1.,
                                off_value=1.)
            temps = tf.unpack(ixtemp, axis=1)

            model_logprob = [l / t for l, t in
                             zip(self.runtime_logits, temps)]

        for p in model_logprob:  # time steps

            # with gather and flattening
            sample_id = tf.stop_gradient(tf.cast(tf.multinomial(p, sample_size),
                                                 tf.int32))  # batch_size x 1
            sample_id = tf.squeeze(sample_id, [1])  # batch_size

            # use dynamic partition to get the logprobs of the samples
            partitions = tf.one_hot(sample_id, voc_size, dtype=tf.int32)
            sample_logprob_word = \
                tf.dynamic_partition(p, partitions=partitions,
                                     num_partitions=2)[1]

            sample_ids.append(sample_id)
            sample_logprob += sample_logprob_word

        return tf.pack(sample_ids), sample_logprob, ix

    # pylint: disable=too-many-branches
    def attention_decoder(
            self,
            go_symbols: tf.Tensor,
            train_inputs: tf.Tensor = None,
            attention_on_input=True,
            train_mode: bool = False,
            scope: Union[str, tf.VariableScope] = None,
            sample_mode: int = 0,
            temperature: float = 1.0,
            store_logits: bool = False,
            store_rnn_outputs: bool = False,
            store_rnn_states: bool = False) -> Tuple[
        List[tf.Tensor], List[tf.Tensor], List[tf.Tensor],
        List[tf.Tensor], List[tf.Tensor], tf.Tensor]:
        """Run the decoder RNN.

        Arguments:
            go_symbols: The tensor of start symbols of shape (1, batch_size)
            train_inputs: Training inputs to feed the decoder with. These are
                not used when `train_mode = False`
            attention_on_input: Flag whether attention from previous time step
                is fed to the input in the next step.
            train_mode: Boolean flag whether the decoder is running in
                train (with ground truth inputs) or runtime mode (with inputs
                decoded using the loop function)
            scope: Variable scope to use
            sample_mode: -k: sampling k times from negative logits,
                k: sampling k times from logits, 0: greedy
            temperature: softmax temperature for sampling
            store_rnn_outputs: whether to return the RNN outputs from every step
            store_rnn_states: whether to return the RNN states from every step
            store_logits: whether to return the logits from every step
        """
        att_objects = [self.get_attention_object(e, train_mode)
                       for e in self.encoders]
        att_objects = [a for a in att_objects if a is not None]

        cell = self._get_rnn_cell()

        unk_id = self.vocabulary.get_unk_id()

        outputs = []
        states = []
        predictions = []
        logprob_predicted = 0
        logits = []

        with tf.variable_scope(scope or "attention_decoder"):
            if self._rnn_cell == 'GRU':
                state = self.initial_state
            elif self._rnn_cell == 'LSTM':
                # pylint: disable=redefined-variable-type
                state = tf.nn.rnn_cell.LSTMStateTuple(
                    self.initial_state, self.initial_state)
                # pylint: enable=redefined-variable-type
            else:
                raise ValueError("Unknown RNN cell.")

            prev = None

            attns = [tf.zeros([self.batch_size, a.attn_size])
                     for a in att_objects]

            # sample index to take negative logits for sampling
            ix = tf.random_uniform((1,), 1, self.max_output_len + 1, tf.int32)

            # modify temperature for this index for negative sampling
            if sample_mode < 0:
                ixtemp = tf.one_hot(ix, self.max_output_len + 1,
                                    on_value=-temperature,
                                    off_value=temperature)
                temps = tf.unpack(ixtemp, axis=1)
            else:
                temps = [temperature] * (self.max_output_len + 1)

            voc_size = len(self.vocabulary)

            for i, temp in zip(range(self.max_output_len + 1), temps):
                if i > 0:
                    tf.get_variable_scope().reuse_variables()

                if prev is None:
                    assert i == 0
                    inp = go_symbols[0]

                elif train_mode:
                    if i < self.max_output_len:
                        inp = train_inputs[i - 1]
                        out_activation = self._logit_function(prev)
                        if store_logits:
                            logits.append(out_activation)
                    else:
                        out_activation = self._logit_function(prev)
                        if store_logits:
                            logits.append(out_activation)
                        break
                else:
                    # during runtime find index for output word by:
                    # 1) greedy decoding (i.e. taking the argmax of logits)
                    # 2) sampling, either from positive or negative logits
                    with tf.variable_scope("loop_function", reuse=True):

                        out_activation = self._logit_function(prev)
                        if store_logits:
                            logits.append(out_activation)

                        if sample_mode == 0:
                            # greedy
                            prev_word_index = tf.cast(tf.argmax(out_activation,
                                                                1), tf.int32)

                        else:
                            # sampling
                            out_activation = tf.scalar_mul(1./temp,
                                                           out_activation)
                            prev_word_index = tf.stop_gradient(tf.cast(
                                tf.multinomial(out_activation, 1),
                                tf.int32))  # batch_size x sample_size
                            prev_word_index = tf.squeeze(prev_word_index, [1])

                        # use dynamic partition to get the logprobs of the
                        # samples
                        partitions = tf.one_hot(prev_word_index, voc_size,
                                                dtype=tf.int32)
                        sample_logprob_word = tf.dynamic_partition(
                            out_activation, partitions=partitions,
                            num_partitions=2)[1]

                        # compute log probabilities from logits
                        z = tf.reduce_logsumexp(out_activation, [1])
                        sample_logprob_word_normalized = sample_logprob_word - z
                        logprob_predicted += sample_logprob_word_normalized

                        inp = self._embed_and_dropout(prev_word_index)

                        # if predicted word is UNK, replace with negative index
                        # of maximally aligned source word
                        def replace_unk(token_id):
                            i, j = tf.unpack(token_id)
                            return tf.cond(tf.equal(i, unk_id), lambda: (-j),
                                           lambda: i)

                        attns_time = [a.attentions_in_time[-1] for
                                      a in att_objects]
                        argmax_attention = tf.cast(tf.argmax(attns_time[0], 1),
                                                   tf.int32)
                        symb_and_att = tf.pack([prev_word_index,
                                                argmax_attention], 1)
                        unk_replaced = tf.map_fn(lambda token_id:
                                                 replace_unk(token_id),
                                                 symb_and_att)

                        predictions.append(unk_replaced)

                if abs(sample_mode) <= 1:

                    # Merge input and previous attentions into one vector of the
                    # right size.
                    if attention_on_input:
                        x = linear([inp] + attns, self.embedding_size)
                    else:
                        x = inp
                    # Run the RNN.

                    cell_output, state = cell(x, state)
                    if store_rnn_states:
                        states.append(state)

                    # Run the attention mechanism.

                    attns = [a.attention(cell_output) for a in att_objects]

                    with tf.name_scope("rnn_output_projection"):
                        if attns:
                            output = linear([cell_output] + attns,
                                            cell.output_size,
                                            scope="AttnOutputProjection")
                        else:
                            output = cell_output

                else:
                    raise NotImplementedError(
                        "Multiple samples are not implemented yet")

                prev = output

                if store_rnn_outputs:
                    outputs.append(output)

        return outputs, states, predictions, logprob_predicted, logits, ix - 1

    def _visualize_attention(self, neg=False):
        """Create image summaries with attentions"""
        att_objects = self._runtime_attention_objects.values()

        for i, a in enumerate(att_objects):
            alignments = tf.expand_dims(tf.transpose(
                tf.pack(a.attentions_in_time), perm=[1, 2, 0]), -1)

            tf.image_summary(
                "attention_{}_{}".format(i, neg), alignments,
                collections=["summary_val_plots"],
                max_images=256)

    def feed_dict(self, dataset: Dataset, train: bool = False) -> FeedDict:
        """Populate the feed dictionary for the decoder object

        Arguments:
            dataset: The dataset to use for the decoder.
            train: Boolean flag, telling whether this is a training run
        """
        sentences = cast(Iterable[List[str]],
                         dataset.get_series(self.data_id, allow_none=True))

        if sentences is None and train:
            raise ValueError("When training, you must feed "
                             "reference sentences")

        sentences_list = list(sentences) if sentences is not None else None

        fd = {self.train_mode: train}  # type: FeedDict

        go_symbol_idx = self.vocabulary.get_word_index(START_TOKEN)
        fd[self.go_symbols] = np.full([1, len(dataset)], go_symbol_idx,
                                      dtype=np.int32)

        if sentences is not None:
            # train_mode=False, since we don't want to <unk>ize target words!
            inputs, weights = self.vocabulary.sentences_to_tensor(
                sentences_list, self.max_output_len, train_mode=False,
                add_start_symbol=False, add_end_symbol=True)

            assert inputs.shape == (self.max_output_len, len(sentences_list))
            assert weights.shape == (self.max_output_len, len(sentences_list))

            fd[self.train_inputs] = inputs
            fd[self.train_padding] = weights

        return fd

    def _get_placeholders(self):
        """
        Get all the placeholders of the decoder
        :return:
        """
        placeholders = [self.rewards, self.epoch, self.go_symbols,
                        self.train_mode, self.train_inputs, self.train_padding,
                        self.baseline, self.step]
        return placeholders
