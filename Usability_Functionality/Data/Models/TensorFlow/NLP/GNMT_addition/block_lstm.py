"""LSTM Block Cell ops."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import tensorflow as tf
#24
try:
    from tensorflow.python.ops import gen_rnn_ops as gen_ops
except ImportError:
    from tensorflow.contrib.rnn.ops import gen_lstm_ops as gen_ops
from tensorflow.python.framework import function
from tensorflow.python.layers import base as base_layer

#18
class LSTMBlockWrapper(base_layer.Layer):

  @abc.abstractproperty
  def num_units(self):
    pass

  @abc.abstractmethod
  def _call_cell(self, inputs, initial_cell_state, initial_output, dtype,
                 sequence_length):
    pass

  def call(self, inputs, initial_state=None, dtype=None, sequence_length=None,
           mask_output=False):
    is_list = isinstance(inputs, list)
    if is_list:
      inputs = tf.stack(inputs)
    inputs_shape = inputs.get_shape().with_rank(3)
    if not inputs_shape[2]:
      raise ValueError("Expecting inputs_shape[2] to be set: %s" % inputs_shape)
    batch_size = inputs_shape[1].value
    if batch_size is None:
      batch_size = tf.shape(inputs)[1]
    time_len = inputs_shape[0].value
    if time_len is None:
      time_len = tf.shape(inputs)[0]

    # Provide default values for initial_state and dtype
    if initial_state is None:
      if dtype is None:
        raise ValueError("Either initial_state or dtype needs to be specified")
      z = tf.zeros(
          tf.stack([batch_size, self.num_units]), dtype=dtype)
      initial_state = z, z
    else:
      if len(initial_state) != 2:
        raise ValueError(
            "Expecting initial_state to be a tuple with length 2 or None")
      if dtype is None:
        dtype = initial_state[0].dtype

    # create the actual cell
    if sequence_length is not None:
      sequence_length = tf.convert_to_tensor(sequence_length)
    initial_cell_state, initial_output = initial_state  # pylint: disable=unpacking-non-sequence
    cell_states, outputs = self._call_cell(
        inputs, initial_cell_state, initial_output, dtype, sequence_length)

    if sequence_length is not None:
      if mask_output:
        # Mask out the part beyond sequence_length.
        # In MLPerf we don't do it b.c output is masked when computing loss.
        # And in inference we don't use this layer.
        mask = tf.transpose(
            tf.sequence_mask(sequence_length, time_len, dtype=dtype),
            [1, 0])
        mask = tf.tile(
            tf.expand_dims(mask, axis=-1), [1, 1, self.num_units])
        outputs *= mask
      # sequence_length can't be zero in our impl, pass sequence_length -1 for
      # indices.
      mod_cell_states = cell_states
      mod_outputs = outputs
      final_cell_state = self._gather_states(mod_cell_states,
                                             sequence_length - 1, batch_size)#2
      final_output = self._gather_states(mod_outputs, sequence_length - 1,
                                         batch_size)#2
    else:
      # No sequence_lengths used: final state is the last state
      final_cell_state = cell_states[-1]
      final_output = outputs[-1]

    if is_list:
      # Input was a list, so return a list
      outputs = tf.unstack(outputs)#1

    final_state = tf.nn.rnn_cell.LSTMStateTuple(final_cell_state, final_output)#1
    return outputs, final_state
  #2
  def _gather_states(self, data, indices, batch_size):
    """Produce `out`, s.t. out(i, j) = data(indices(i), i, j)."""
    gather_indices = tf.stack([indices, tf.range(batch_size)], axis=1)
    # TODO(jamesqin): ScatterNd doesn't support fp16 on GPU.
    return tf.gather_nd(data, gather_indices)

#6
class LSTMBlockFusedCell(LSTMBlockWrapper):
  def __init__(self,
               num_units,
               forget_bias=1.0,
               cell_clip=None,
               use_peephole=False,
               reuse=None,
               dtype=None,
               name="lstm_cell"):
    super(LSTMBlockFusedCell, self).__init__(
        _reuse=reuse, name=name, dtype=dtype)
    self._num_units = num_units
    self._forget_bias = forget_bias
    self._cell_clip = cell_clip if cell_clip is not None else -1
    self._use_peephole = use_peephole

    # Inputs must be 3-dimensional.
    self.input_spec = base_layer.InputSpec(ndim=3)

  @property
  def num_units(self):
    return self._num_units

  def build(self, input_shape):
    input_size = input_shape[2].value
    self._kernel = self.add_variable(
        "kernel", [input_size + self._num_units, self._num_units * 4])
    self._bias = self.add_variable(
        "bias", [self._num_units * 4],
        initializer=tf.constant_initializer(0.0))
    if self._use_peephole:
      self._w_i_diag = self.add_variable("w_i_diag", [self._num_units])
      self._w_f_diag = self.add_variable("w_f_diag", [self._num_units])
      self._w_o_diag = self.add_variable("w_o_diag", [self._num_units])

    self.built = True

  def _call_cell(self,
                 inputs,
                 initial_cell_state=None,
                 initial_output=None,
                 dtype=None,
                 sequence_length=None):

    inputs_shape = inputs.get_shape().with_rank(3)
    time_len = inputs_shape[0].value
    if time_len is None:
      time_len = tf.shape(inputs)[0]

    if self._use_peephole:
      wci = self._w_i_diag
      wco = self._w_o_diag
      wcf = self._w_f_diag
    else:
      wci = wcf = wco = tf.zeros([self._num_units], dtype=dtype)

    if sequence_length is None:
      max_seq_len = tf.to_int64(time_len)
    else:
      max_seq_len = tf.to_int64(tf.reduce_max(sequence_length))

    _, cs, _, _, _, _, h = gen_ops.block_lstm(
        seq_len_max=max_seq_len,
        x=inputs,
        cs_prev=initial_cell_state,
        h_prev=initial_output,
        w=self._kernel,
        wci=wci,
        wcf=wcf,
        wco=wco,
        b=self._bias,
        forget_bias=self._forget_bias,
        cell_clip=self._cell_clip,
        use_peephole=self._use_peephole)
    return cs, h
