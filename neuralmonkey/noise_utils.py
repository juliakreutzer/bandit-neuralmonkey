from typing import cast, Iterable, List, Callable, Optional, Union, Any, \
    Tuple, Dict

import tensorflow as tf
from neuralmonkey.nn.projection import linear
from tensorflow.python.util import nest
from tensorflow.python.framework import dtypes
from neuralmonkey.nn.init_ops import orthogonal_initializer



def _rnn_step(
    time, sequence_length, min_sequence_length, max_sequence_length,
    zero_output, state, call_cell, state_size, skip_conditionals=False):

  # Convert state to a list for ease of use
  flat_state = nest.flatten(state)
  flat_zero_output = nest.flatten(zero_output)

  def _copy_one_through(output, new_output):
    copy_cond = (time >= sequence_length)
    return tf.select(copy_cond, output, new_output)

  def _copy_some_through(flat_new_output, flat_new_state):
    # Use broadcasting select to determine which values should get
    # the previous state & zero output, and which values should get
    # a calculated state & output.
    flat_new_output = [
        _copy_one_through(zero_output, new_output)
        for zero_output, new_output in zip(flat_zero_output, flat_new_output)]
    flat_new_state = [
        _copy_one_through(state, new_state)
        for state, new_state in zip(flat_state, flat_new_state)]
    return flat_new_output + flat_new_state

  def _maybe_copy_some_through():
    """Run RNN step.  Pass through either no or some past state."""
    new_output, new_state = call_cell()

    nest.assert_same_structure(state, new_state)

    flat_new_state = nest.flatten(new_state)
    flat_new_output = nest.flatten(new_output)
    return tf.cond(
        # if t < min_seq_len: calculate and return everything
        time < min_sequence_length, lambda: flat_new_output + flat_new_state,
        # else copy some of it through
        lambda: _copy_some_through(flat_new_output, flat_new_state))

  # TODO(ebrevdo): skipping these conditionals may cause a slowdown,
  # but benefits from removing cond() and its gradient.  We should
  # profile with and without this switch here.
  if skip_conditionals:
    # Instead of using conditionals, perform the selective copy at all time
    # steps.  This is faster when max_seq_len is equal to the number of unrolls
    # (which is typical for dynamic_rnn).
    new_output, new_state = call_cell()
    nest.assert_same_structure(state, new_state)
    new_state = nest.flatten(new_state)
    new_output = nest.flatten(new_output)
    final_output_and_state = _copy_some_through(new_output, new_state)
  else:
    empty_update = lambda: flat_zero_output + flat_state
    final_output_and_state = tf.cond(
        # if t >= max_seq_len: copy all state through, output zeros
        time >= max_sequence_length, empty_update,
        # otherwise calculation is required: copy some or all of it through
        _maybe_copy_some_through)

  if len(final_output_and_state) != len(flat_zero_output) + len(flat_state):
    raise ValueError("Internal error: state and output were not concatenated "
                     "correctly.")
  final_output = final_output_and_state[:len(flat_zero_output)]
  final_state = final_output_and_state[len(flat_zero_output):]

  for output, flat_output in zip(final_output, flat_zero_output):
    output.set_shape(flat_output.get_shape())
  for substate, flat_substate in zip(final_state, flat_state):
    substate.set_shape(flat_substate.get_shape())

  final_output = nest.pack_sequence_as(
      structure=zero_output, flat_sequence=final_output)
  final_state = nest.pack_sequence_as(
      structure=state, flat_sequence=final_state)

  return final_output, final_state


def noisy_dynamic_rnn(cell, inputs, sequence_length=None, initial_state=None,
                dtype=None, parallel_iterations=None, swap_memory=False,
                time_major=False, scope=None, train_mode=False, direction="fw"):
    flat_input = nest.flatten(inputs)

    if not time_major:
        # (B,T,D) => (T,B,D)
        flat_input = tuple(tf.transpose(input_, [1, 0, 2])
                           for input_ in flat_input)

    parallel_iterations = parallel_iterations or 32
    if sequence_length is not None:
        sequence_length = tf.to_int32(sequence_length)
        sequence_length = tf.identity(  # Just to find it in the graph.
            sequence_length, name="sequence_length")

    # Create a new scope in which the caching device is either
    # determined by the parent scope, or is set to place the cached
    # Variable using the same placement as for the rest of the RNN.
    with tf.variable_scope(scope or "RNN") as varscope:
        if varscope.caching_device is None:
            varscope.set_caching_device(lambda op: op.device)
        input_shape = tuple(tf.shape(input_) for input_ in flat_input)
        batch_size = input_shape[0][1]

        for input_ in input_shape:
            if input_[1].get_shape() != batch_size.get_shape():
                raise ValueError("All inputs should have the same batch size")

        if initial_state is not None:
            state = initial_state
        else:
            if not dtype:
                raise ValueError(
                    "If no initial_state is provided, dtype must be.")
            state = cell.zero_state(batch_size, dtype)

        def _assert_has_shape(x, shape):
            x_shape = tf.shape(x)
            packed_shape = tf.pack(shape)
            return tf.Assert(
                tf.reduce_all(tf.equal(x_shape, packed_shape)),
                ["Expected shape for Tensor %s is " % x.name,
                 packed_shape, " but saw shape: ", x_shape])

        if sequence_length is not None:
            # Perform some shape validation
            with tf.control_dependencies(
                    [_assert_has_shape(sequence_length, [batch_size])]):
                sequence_length = tf.identity(
                    sequence_length, name="CheckSeqLen")

        inputs = nest.pack_sequence_as(structure=inputs,
                                       flat_sequence=flat_input)

        (outputs, final_state, gradient) = noisy_dynamic_rnn_loop(
            cell,
            inputs,
            state,
            parallel_iterations=parallel_iterations,
            swap_memory=swap_memory,
            sequence_length=sequence_length,
            dtype=dtype, train_mode=train_mode, direction=direction)

        # Outputs of _dynamic_rnn_loop are always shaped [time, batch, depth].
        # If we are performing batch-major calculations, transpose output back
        # to shape [batch, time, depth]
        if not time_major:
            # (T,B,D) => (B,T,D)
            flat_output = nest.flatten(outputs)
            flat_output = [tf.transpose(output, [1, 0, 2])
                           for output in flat_output]
            outputs = nest.pack_sequence_as(
                structure=outputs, flat_sequence=flat_output)

        return (outputs, final_state, gradient)


def noisy_dynamic_rnn_loop(cell,
                           inputs,
                           initial_state,
                           parallel_iterations,
                           swap_memory,
                           sequence_length=None,
                           dtype=None, train_mode=False, direction="fw"):
    state = initial_state
    assert isinstance(parallel_iterations,
                      int), "parallel_iterations must be int"

    state_size = cell.state_size

    flat_input = nest.flatten(inputs)
    flat_output_size = nest.flatten(cell.output_size)

    # Construct an initial output
    input_shape = tf.shape(flat_input[0])
    time_steps = input_shape[0]
    batch_size = input_shape[1]

    inputs_got_shape = tuple(input_.get_shape().with_rank_at_least(3)
                             for input_ in flat_input)

    const_time_steps, const_batch_size = inputs_got_shape[0].as_list()[:2]

    for shape in inputs_got_shape:
        if not shape[2:].is_fully_defined():
            raise ValueError(
                "Input size (depth of inputs) must be accessible via shape inference,"
                " but saw value None.")
        got_time_steps = shape[0]
        got_batch_size = shape[1]
        if const_time_steps != got_time_steps:
            raise ValueError(
                "Time steps is not the same for all the elements in the input in a "
                "batch.")
        if const_batch_size != got_batch_size:
            raise ValueError(
                "Batch_size is not the same for all the elements in the input.")

    # Prepare dynamic conditional copying of state & output
    def _create_zero_arrays(size):
        size = tf.nn.rnn_cell._state_size_with_prefix(size, prefix=[batch_size])
        return tf.zeros(
            tf.pack(size), _infer_state_dtype(dtype, state))

    flat_zero_output = tuple(_create_zero_arrays(output)
                             for output in flat_output_size)
    zero_output = nest.pack_sequence_as(structure=cell.output_size,
                                        flat_sequence=flat_zero_output)

    if sequence_length is not None:
        min_sequence_length = tf.reduce_min(sequence_length)
        max_sequence_length = tf.reduce_max(sequence_length)

    time = tf.constant(0, dtype=dtypes.int32, name="time")

    with tf.name_scope("dynamic_rnn") as scope:
        base_name = scope

    def _create_ta(name, dtype):
        return tf.TensorArray(dtype=dtype,
                                            size=time_steps,
                                            tensor_array_name=base_name + name)

    output_ta = tuple(_create_ta("output_%d" % i,
                                 _infer_state_dtype(dtype, state))
                      for i in range(len(flat_output_size)))
    input_ta = tuple(_create_ta("input_%d" % i, flat_input[0].dtype)
                     for i in range(len(flat_input)))

    input_ta = tuple(ta.unpack(input_)
                     for ta, input_ in zip(input_ta, flat_input))

    # TODO how to measure moments? variables are not initialized yet, but noise matrix has to be defined
    if train_mode:
        noise_dist = tf.contrib.distributions.Normal(mu=0., sigma=1.)
        max_input_len, batch_size, embedding_size = flat_input[0].get_shape().as_list()

        # for gates
        noise_shape_gates = [embedding_size+state_size, 2 * state_size]
        noise_matrix_gates = tf.Variable(tf.zeros(noise_shape_gates), name=direction+"_noise_gates",
                                   trainable=False)
        # only sample once per batch and use this noise for the whole sequence
        noise_matrix_gates = noise_matrix_gates.assign(noise_dist.sample(noise_shape_gates))
        # noise_matrix = tf.nn.l2_normalize(noise_matrix, [0, 1])

        # for candidate
        noise_shape_candidate = [embedding_size + state_size, state_size]
        noise_matrix_candidate = tf.Variable(tf.zeros(noise_shape_candidate),
                                         name=direction + "_noise_candidate",
                                         trainable=False)
        # only sample once per batch and use this noise for the whole sequence
        noise_matrix_candidate = noise_matrix_candidate.assign(
            noise_dist.sample(noise_shape_candidate))

        noise_matrix = noise_matrix_gates, noise_matrix_candidate

    else:
        noise_matrix = None, None

    def _time_step(time, output_ta_t, state):
        """Take a time step of the dynamic RNN.

        Args:
          time: int32 scalar Tensor.
          output_ta_t: List of `TensorArray`s that represent the output.
          state: nested tuple of vector tensors that represent the state.

        Returns:
          The tuple (time + 1, output_ta_t with updated flow, new_state).
        """

        input_t = tuple(ta.read(time) for ta in input_ta)
        # Restore some shape information
        for input_, shape in zip(input_t, inputs_got_shape):
            input_.set_shape(shape[1:])

        input_t = nest.pack_sequence_as(structure=inputs, flat_sequence=input_t)
        if train_mode:
            call_cell = lambda: cell(input_t, state, noise_recurrent=noise_matrix)
        else:
            call_cell = lambda: cell(input_t, state)

        if sequence_length is not None:
            step_output = _rnn_step(
                time=time,
                sequence_length=sequence_length,
                min_sequence_length=min_sequence_length,
                max_sequence_length=max_sequence_length,
                zero_output=zero_output,
                state=state,
                call_cell=call_cell,
                state_size=state_size,
                skip_conditionals=True)

            (output, new_state) = step_output

        else:
            step_output = call_cell()
            (output, new_state) = step_output

        # Pack state if using state tuples
        output = nest.flatten(output)

        output_ta_t = tuple(
            ta.write(time, out) for ta, out in zip(output_ta_t, output))

        return (time + 1, output_ta_t, new_state)

    _, output_final_ta, final_state = tf.while_loop(
        cond=lambda time, *_: time < time_steps,
        body=_time_step,
        loop_vars=(time, output_ta, state),
        parallel_iterations=parallel_iterations,
        swap_memory=swap_memory)

    # get the vars for gradient here
    # not in while because context is limited and has to keep same shape
    tf.get_variable_scope().reuse_variables()

    # mean, var = tf.nn.moments(tf.get_variable("Linear/Matrix",
    #                             shape=[input_shape+state_shape,
    #                                    2*self._num_units]),
    #                          axes=[0, 1])
    # batch-normalize noise according to mean and var of matrix
    # noise_recurrent = tf.nn.batch_normalization(
    #    noise_recurrent, mean=mean, variance=var, offset=None,
    #    scale=None, variance_epsilon=1.0e-5)

    # gradient_val = tf.concat(0, [noise_empty, noise_recurrent])
    gradient_val_gates, gradient_val_candidate = noise_matrix
    gradient = [(gradient_val_gates,
                 tf.get_variable("OrthoGRUCell/Gates/Linear/Matrix",
                                 validate_shape=False)),
                (gradient_val_candidate,
                tf.get_variable("OrthoGRUCell/Candidate/Linear/Matrix",
                                validate_shape=False))]

    # Unpack final output if not using output tuples.
    final_outputs = tuple(ta.pack() for ta in output_final_ta)

    # Restore some shape information
    for output, output_size in zip(final_outputs, flat_output_size):
        shape = tf.nn.rnn_cell._state_size_with_prefix(
            output_size, prefix=[const_time_steps, const_batch_size])
        output.set_shape(shape)

    final_outputs = nest.pack_sequence_as(
        structure=cell.output_size, flat_sequence=final_outputs)

    return (final_outputs, final_state, gradient)

def _infer_state_dtype(explicit_dtype, state):
  """Infer the dtype of an RNN state.

  Args:
    explicit_dtype: explicitly declared dtype or None.
    state: RNN's hidden state. Must be a Tensor or a nested iterable containing
      Tensors.

  Returns:
    dtype: inferred dtype of hidden state.

  Raises:
    ValueError: if `state` has heterogeneous dtypes or is empty.
  """
  if explicit_dtype is not None:
    return explicit_dtype
  elif nest.is_sequence(state):
    inferred_dtypes = [element.dtype for element in nest.flatten(state)]
    if not inferred_dtypes:
      raise ValueError("Unable to infer dtype from empty state.")
    all_same = all([x == inferred_dtypes[0] for x in inferred_dtypes])
    if not all_same:
      raise ValueError(
          "State has tensors of different inferred_dtypes. Unable to infer a "
          "single representative dtype.")
    return inferred_dtypes[0]
  else:
    return state.dtype


class NoisyGRUCell(tf.nn.rnn_cell.RNNCell):
    """
    Gated Recurrent Unit cell with noise in recursion function.
    It is based on the TensorFlow implementatin of GRU.
    """

    def __init__(self, num_units: int) -> None:
        self._num_units = num_units

    @property
    def output_size(self):
        return self._num_units

    @property
    def state_size(self):
        return self._num_units

    def __call__(self, inputs, state, noise_recurrent=None,
                 scope=None, delta=1.0) -> Tuple[tf.Tensor, tf.Tensor]:
        """Gated recurrent unit (GRU) with nunits cells."""

        with tf.variable_scope(scope or "GRUCell"):
            with tf.variable_scope("Gates"):  # Reset gate and update gate.
                #input_shape = inputs.get_shape().as_list()[1]
                #state_shape = self._num_units
                linear_transform = linear([inputs, state], 2 * self._num_units,
                                          scope=tf.get_variable_scope(),
                                          bias=True)
                #gradient = None
                if noise_recurrent[0] is not None:
                    gate_noise = noise_recurrent[0]
                    # first part of noise matrix that transforms input is empty
                    #noise_empty = tf.zeros([input_shape, 2*self._num_units])
                    # second part is sampled from Gaussian
                    #tf.get_variable_scope().reuse_variables()

                    #mean, var = tf.nn.moments(tf.get_variable("Linear/Matrix",
                    #                             shape=[input_shape+state_shape,
                    #                                    2*self._num_units]),
                    #                          axes=[0, 1])
                    # batch-normalize noise according to mean and var of matrix
                    #noise_recurrent = tf.nn.batch_normalization(
                    #    noise_recurrent, mean=mean, variance=var, offset=None,
                    #    scale=None, variance_epsilon=1.0e-5)

                    # noise to both recurrent matrices for gate computation
                    input_and_state = tf.concat(1, [inputs, state])
                    to_add = tf.batch_matmul(input_and_state, gate_noise)
                    linear_transform += delta*to_add


                # We start with bias of 1.0 to not reset and not update.
                r, u = tf.split(1, 2, linear_transform)
                # inputs: batch x embedding_size
                # state: batch x rnn_size
                # r: batch x rnn_size

                r, u = tf.sigmoid(r), tf.sigmoid(u)


            with tf.variable_scope("Candidate"):
                linear_transform = linear([inputs, r * state],
                                   self._num_units, scope=tf.get_variable_scope(), bias=True)

                if noise_recurrent[1] is not None:
                    candidate_noise = noise_recurrent[1]
                    input_and_state_r = tf.concat(1, [inputs, r*state])
                    to_add = tf.batch_matmul(input_and_state_r, candidate_noise)
                    linear_transform += delta*to_add

                c = tf.tanh(linear_transform)

            new_h = u * state + (1 - u) * c  # batch_size x rnn_size
            # TODO noise in candidate computation?
            return new_h, new_h

            # Wx + Uh_t-1 -> W[x, h_t-1]
            # plus noise: Wx + (U+noise)h_t-1 -> Wx + Uh_t-1 + noise*h_t-1 -> W[x, h_t-1] + noise*h_t-1


class NoisyOrthoGRUCell(NoisyGRUCell):
    """Classic GRU cell but initialized using random orthogonal matrices"""

    def __call__(self, inputs, state, noise_recurrent=None, scope=None, delta=1.0):
        with tf.variable_scope(scope or "OrthoGRUCell") as vscope:
            vscope.set_initializer(orthogonal_initializer())
            return super().__call__(inputs, state, noise_recurrent=noise_recurrent, scope=vscope, delta=delta)