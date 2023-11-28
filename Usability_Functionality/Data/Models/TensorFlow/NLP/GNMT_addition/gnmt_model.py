from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tensorflow.contrib.cudnn_rnn.python.layers import cudnn_rnn
import attention_wrapper
import block_lstm
import model
import model_helper
from utils import misc_utils as utils

#411
class GNMTModel(model.BaseModel):
  #1
  def __init__(self,
               hparams,
               mode,
               features,
               scope=None,
               extra_args=None):
    self.is_gnmt_attention = (
        hparams.attention_architecture in ["gnmt", "gnmt_v2"])

    super(GNMTModel, self).__init__(
        hparams=hparams,
        mode=mode,
        features=features,
        scope=scope,
        extra_args=extra_args)
  #3
  def _prepare_beam_search_decoder_inputs(
      self, beam_width, memory, source_sequence_length, encoder_state):
    memory = tf.contrib.seq2seq.tile_batch(
        memory, multiplier=beam_width)
    source_sequence_length = tf.contrib.seq2seq.tile_batch(
        source_sequence_length, multiplier=beam_width)
    encoder_state = tf.contrib.seq2seq.tile_batch(
        encoder_state, multiplier=beam_width)
    batch_size = self.batch_size * beam_width
    return memory, source_sequence_length, encoder_state, batch_size

  def _build_encoder(self, hparams):
    """Build a GNMT encoder."""
    assert hparams.encoder_type == "gnmt"

    # Build GNMT encoder.
    num_bi_layers = 1
    num_uni_layers = self.num_encoder_layers - num_bi_layers
    utils.print_out("# Build a GNMT encoder")
    utils.print_out("  num_bi_layers = %d" % num_bi_layers)
    utils.print_out("  num_uni_layers = %d" % num_uni_layers)

    # source is batch-majored
    source = self.features["source"]
    import sys
    print('source.shape: %s' % source.shape, file=sys.stderr)
    if self.time_major:
      # Later rnn would use time-majored inputs
      source = tf.transpose(source)

    with tf.variable_scope("encoder"):
      dtype = self.dtype

      encoder_emb_inp = tf.cast(
          self.encoder_emb_lookup_fn(self.embedding_encoder, source), dtype)

      # Build 1st bidi layer.
      bi_encoder_outputs, bi_encoder_state = self._build_encoder_layers_bidi(
          encoder_emb_inp, self.features["source_sequence_length"], hparams,
          dtype)

      # Build all the rest unidi layers
      encoder_state, encoder_outputs = self._build_encoder_layers_unidi(
          bi_encoder_outputs, self.features["source_sequence_length"],
          num_uni_layers, hparams, dtype)

      # Pass all encoder states to the decoder
      #   except the first bi-directional layer
      encoder_state = (bi_encoder_state[1],) + (
          (encoder_state,) if num_uni_layers == 1 else encoder_state)
    return encoder_outputs, encoder_state
  #23+44+11 = 78
  def _build_encoder_layers_bidi(self, inputs, sequence_length, hparams, dtype):
    """docstring."""
    if hparams.use_fused_lstm:
      fn = self._build_bidi_rnn_fused#23
    elif hparams.use_cudnn_lstm:#44
      fn = self._build_bidi_rnn_cudnn
    else:
      fn = self._build_bidi_rnn_base#11
    return fn(inputs, sequence_length, hparams, dtype)
  #23
  def _build_bidi_rnn_fused(self, inputs, sequence_length, hparams, dtype):
    if (not np.isclose(hparams.dropout, 0.) and
        self.mode == tf.contrib.learn.ModeKeys.TRAIN):
      inputs = tf.nn.dropout(inputs, keep_prob=1-hparams.dropout)

    fwd_cell = block_lstm.LSTMBlockFusedCell(#6
        hparams.num_units, hparams.forget_bias, dtype=dtype)
    fwd_encoder_outputs, (fwd_final_c, fwd_final_h) = fwd_cell(
        inputs,
        dtype=dtype,
        sequence_length=sequence_length)

    inputs_r = tf.reverse_sequence(#1
        inputs, sequence_length, batch_axis=1, seq_axis=0)
    bak_cell = block_lstm.LSTMBlockFusedCell(#6
        hparams.num_units, hparams.forget_bias, dtype=dtype)
    bak_encoder_outputs, (bak_final_c, bak_final_h) = bak_cell(#6
        inputs_r,
        dtype=dtype,
        sequence_length=sequence_length)
    bak_encoder_outputs = tf.reverse_sequence(#1
        bak_encoder_outputs, sequence_length, batch_axis=1, seq_axis=0)
    bi_encoder_outputs = tf.concat(#1
        [fwd_encoder_outputs, bak_encoder_outputs], axis=-1)
    fwd_state = tf.nn.rnn_cell.LSTMStateTuple(fwd_final_c, fwd_final_h)#1
    bak_state = tf.nn.rnn_cell.LSTMStateTuple(bak_final_c, bak_final_h)#1
    bi_encoder_state = (fwd_state, bak_state)

    # mask aren't applied on outputs, but final states are post-masking.
    return bi_encoder_outputs, bi_encoder_state
  #13
  def _build_unidi_rnn_fused(self, inputs, state,
                             sequence_length, hparams, dtype):
    if (not np.isclose(hparams.dropout, 0.) and
        self.mode == tf.contrib.learn.ModeKeys.TRAIN):#1
      inputs = tf.nn.dropout(inputs, keep_prob=1-hparams.dropout)#1

    cell = block_lstm.LSTMBlockFusedCell(#6
        hparams.num_units, hparams.forget_bias, dtype=dtype)
    outputs, (final_c, final_h) = cell(#6
        inputs,
        state,
        dtype=dtype,
        sequence_length=sequence_length)

    # mask aren't applied on outputs, but final states are post-masking.
    return outputs, tf.nn.rnn_cell.LSTMStateTuple(final_c, final_h)#1
  #15
  def _build_unidi_rnn_cudnn(self, inputs, state, sequence_length, dtype,
                             hparams, num_layers, is_fwd):
    # cudnn inputs only support time-major
    if not self.time_major:
      inputs = tf.transpose(inputs, axis=[1, 0, 2])#1

    if num_layers == 1 and not np.isclose(hparams.dropout, 0.):
      # Special case when drop is used and only one layer
      dropout = 0.
      inputs = tf.nn.dropout(inputs, keep_prob=1-dropout)#1
    else:
      dropout = hparams.dropout#1

    # the outputs would be in time-majored
    sequence_length = tf.transpose(sequence_length)#1

    if not is_fwd:
      inputs = tf.reverse_sequence(#1
          inputs, sequence_length, batch_axis=1, seq_axis=0)
    cell = tf.contrib.cudnn_rnn.CudnnLSTM(#1
        num_layers=num_layers,
        num_units=hparams.num_units,
        direction=cudnn_rnn.CUDNN_RNN_UNIDIRECTION,
        dtype=self.dtype,
        dropout=dropout)
    outputs, (h, c) = cell(inputs, initial_state=state)

    if not is_fwd:
      outputs = tf.reverse_sequence(#1
          inputs, sequence_length, batch_axis=1, seq_axis=0)
    # NOTICE! There's no way to get the "correct" masked cell state in cudnn
    # rnn.
    if num_layers == 1:
      h = tf.squeeze(h, axis=0)#1
      c = tf.squeeze(c, axis=0)#1
      return outputs, tf.nn.rnn_cell.LSTMStateTuple(c=c, h=h)#1

    # Split h and c to form a
    h.set_shape((num_layers, None, hparams.num_units))#1
    c.set_shape((num_layers, None, hparams.num_units))#1
    hs = tf.unstack(h)#1
    cs = tf.unstack(c)#1
    # The cell passed to bidi-dyanmic-rnn is a MultiRNNCell consisting 2 regular
    # LSTM, the state of each is a simple LSTMStateTuple. Thus the state of the
    # MultiRNNCell is a tuple of LSTMStateTuple.
    states = tuple(#1
        tf.nn.rnn_cell.LSTMStateTuple(c=c, h=h) for h, c in zip(hs, cs))
    # No need to transpose back
    return outputs, states
  #4
  def _build_encoder_cell(self, hparams, num_layers, num_residual_layers,
                          dtype=None):
    """Build a multi-layer RNN cell that can be used by encoder."""
    return model_helper.create_rnn_cell(#4
        unit_type=hparams.unit_type,
        num_units=self.num_units,
        num_layers=num_layers,
        num_residual_layers=num_residual_layers,
        forget_bias=hparams.forget_bias,
        dropout=hparams.dropout,
        mode=self.mode,
        dtype=dtype,
        single_cell_fn=self.single_cell_fn,
        use_block_lstm=hparams.use_block_lstm)
  #11
  def _build_bidi_rnn_base(self, inputs, sequence_length, hparams, dtype):
    """Create and call biddirectional RNN cells."""
    # num_residual_layers: Number of residual layers from top to bottom. For
    # example, if `num_bi_layers=4` and `num_residual_layers=2`, the last 2
    # RNN layers in each RNN cell will be wrapped with `ResidualWrapper`.

    # Construct forward and backward cells
    fw_cell = self._build_encoder_cell(hparams,#4
                                       1,  # num_bi_layers,
                                       0,  # num_bi_residual_layers,
                                       dtype)
    bw_cell = self._build_encoder_cell(hparams,#4
                                       1,  # num_bi_layers,
                                       0,  # num_bi_residual_layers,
                                       dtype)
    if hparams.use_dynamic_rnn:#1
      bi_outputs, bi_state = tf.nn.bidirectional_dynamic_rnn(
          fw_cell,
          bw_cell,
          inputs,
          dtype=dtype,
          sequence_length=sequence_length,
          time_major=self.time_major,
          swap_memory=True)
    else:#1
      bi_outputs, bi_state = tf.contrib.recurrent.bidirectional_functional_rnn(
          fw_cell,
          bw_cell,
          inputs,
          dtype=dtype,
          sequence_length=sequence_length,
          time_major=self.time_major,
          use_tpu=False)
    return tf.concat(bi_outputs, -1), bi_state#1
  #44
  def _build_bidi_rnn_cudnn(self, inputs, sequence_length, hparams, dtype):
    # Notice cudnn rnn dropout is applied between layers. (if 1 layer only then
    # no dropout).
    if not np.isclose(hparams.dropout, 0.):
      inputs = tf.nn.dropout(inputs, keep_prob=1-hparams.dropout)#1
    if not hparams.use_loose_bidi_cudnn_lstm:#1
      fwd_outputs, fwd_states = self._build_unidi_rnn_cudnn(#15
          inputs, None,  # initial_state
          sequence_length, dtype, hparams,
          1,  # num_layer
          is_fwd=True)
      bak_outputs, bak_states = self._build_unidi_rnn_cudnn(#15
          inputs, None,  # initial_state
          sequence_length, dtype, hparams,
          1,  # num_layer
          is_fwd=False)
      bi_outputs = tf.concat([fwd_outputs, bak_outputs], axis=-1)#1
      return bi_outputs, (fwd_states, bak_states)
    else:
      # Cudnn only accept time-majored inputs
      if not self.time_major:
        inputs = tf.transpose(inputs, axis=[1, 0, 2])#1
      bi_outputs, (bi_h, bi_c) = tf.contrib.cudnn_rnn.CudnnLSTM(#1
          num_layers=1,  # num_bi_layers,
          num_units=hparams.num_units,#11
          direction=cudnn_rnn.CUDNN_RNN_BIDIRECTION,#1
          dropout=0.,  # one layer, dropout isn't applied anyway,
          seed=hparams.random_seed,
          dtype=self.dtype,
          kernel_initializer=tf.get_variable_scope().initializer,#1
          bias_initializer=tf.zeros_initializer())(inputs)#1
      # state shape is [num_layers * num_dir, batch, dim]
      bi_h.set_shape((2, None, hparams.num_units))#1
      bi_c.set_shape((2, None, hparams.num_units))#1
      fwd_h, bak_h = tf.unstack(bi_h)#1
      fwd_c, bak_c = tf.unstack(bi_c)#
      # No need to transpose back
      return bi_outputs, (tf.nn.rnn_cell.LSTMStateTuple(c=fwd_c, h=fwd_h),#2
                          tf.nn.rnn_cell.LSTMStateTuple(c=bak_c, h=bak_h))
  #40
  def _build_encoder_layers_unidi(self, inputs, sequence_length,
                                  num_uni_layers, hparams, dtype):
    """Build encoder layers all at once."""
    encoder_outputs = None
    encoder_state = tuple()

    if hparams.use_fused_lstm:
      for i in range(num_uni_layers):
        if (not np.isclose(hparams.dropout, 0.) and
            self.mode == tf.contrib.learn.ModeKeys.TRAIN):#1
          cell_inputs = tf.nn.dropout(inputs, keep_prob=1-hparams.dropout)#1
        else:
          cell_inputs = inputs

        cell = block_lstm.LSTMBlockFusedCell(#6
            hparams.num_units, hparams.forget_bias, dtype=dtype)
        encoder_outputs, (final_c, final_h) = cell(
            cell_inputs,
            dtype=dtype,
            sequence_length=sequence_length)
        encoder_state += (tf.nn.rnn_cell.LSTMStateTuple(final_c, final_h),)#1
        if i >= num_uni_layers - self.num_encoder_residual_layers:
          # Add the pre-dropout inputs. Residual wrapper is applied after
          # dropout wrapper.
          encoder_outputs += inputs
        inputs = encoder_outputs
    elif hparams.use_cudnn_lstm:#1
      # Single layer cudnn rnn, dropout isnt applied in the kernel
      for i in range(num_uni_layers):#8
        if (not np.isclose(hparams.dropout, 0.) and
            self.mode == tf.contrib.learn.ModeKeys.TRAIN):
          inputs = tf.nn.dropout(inputs, keep_prob=1-hparams.dropout)

        encoder_outputs, encoder_states = self._build_unidi_rnn_cudnn(#15
            inputs,
            None,  # initial_state
            sequence_length,
            dtype,
            hparams,
            1,  # num_layer
            is_fwd=True)
        encoder_state += (tf.nn.rnn_cell.LSTMStateTuple(encoder_states.c,
                                                        encoder_states.h),)#1
        if i >= num_uni_layers - self.num_encoder_residual_layers:
          encoder_outputs += inputs
        inputs = encoder_outputs
    else:
      uni_cell = model_helper.create_rnn_cell(#4
          unit_type=hparams.unit_type,
          num_units=hparams.num_units,
          num_layers=num_uni_layers,
          num_residual_layers=self.num_encoder_residual_layers,
          forget_bias=hparams.forget_bias,
          dropout=hparams.dropout,
          dtype=dtype,
          mode=self.mode,
          single_cell_fn=self.single_cell_fn,
          use_block_lstm=hparams.use_block_lstm)

      if hparams.use_dynamic_rnn:
        encoder_outputs, encoder_state = tf.nn.dynamic_rnn(#1
            uni_cell,
            inputs,
            dtype=dtype,
            sequence_length=sequence_length,
            time_major=self.time_major)
      else:
        encoder_outputs, encoder_state = tf.contrib.recurrent.functional_rnn(#1
            uni_cell,
            inputs,
            dtype=dtype,
            sequence_length=sequence_length,
            time_major=self.time_major,
            use_tpu=False)

    return encoder_state, encoder_outputs
  #174
  def _build_decoder_cell(self, hparams, encoder_outputs, encoder_state,
                          source_sequence_length):
    """Build a RNN cell with GNMT attention architecture."""
    # GNMT attention
    assert self.is_gnmt_attention
    attention_option = hparams.attention
    attention_architecture = hparams.attention_architecture
    assert attention_option == "normed_bahdanau"
    assert attention_architecture == "gnmt_v2"

    num_units = hparams.num_units
    infer_mode = hparams.infer_mode
    dtype = tf.float16 if hparams.use_fp16 else tf.float32#1

    if self.time_major:
      memory = tf.transpose(encoder_outputs, [1, 0, 2])#1
    else:
      memory = encoder_outputs

    if (self.mode == tf.contrib.learn.ModeKeys.INFER and#1
        infer_mode == "beam_search"):
      memory, source_sequence_length, encoder_state, batch_size = (
          self._prepare_beam_search_decoder_inputs(#3
              hparams.beam_width, memory, source_sequence_length,
              encoder_state))
    else:
      batch_size = self.batch_size

    attention_mechanism = model.create_attention_mechanism(#122
        num_units, memory, source_sequence_length, dtype=dtype)

    cell_list = model_helper._cell_list(  # pylint: disable=protected-access
        unit_type=hparams.unit_type,#4
        num_units=num_units,
        num_layers=self.num_decoder_layers,
        num_residual_layers=self.num_decoder_residual_layers,
        forget_bias=hparams.forget_bias,
        dropout=hparams.dropout,
        mode=self.mode,
        dtype=dtype,
        single_cell_fn=self.single_cell_fn,
        residual_fn=gnmt_residual_fn,
        use_block_lstm=hparams.use_block_lstm)

    # Only wrap the bottom layer with the attention mechanism.
    attention_cell = cell_list.pop(0)

    # Only generate alignment in greedy INFER mode.
    alignment_history = (self.mode == tf.contrib.learn.ModeKeys.INFER and
                         infer_mode != "beam_search")#1
    attention_cell = attention_wrapper.AttentionWrapper(#30
        attention_cell,
        attention_mechanism,
        attention_layer_size=None,  # don't use attention layer.
        output_attention=False,
        alignment_history=alignment_history,
        name="attention")
    cell = GNMTAttentionMultiCell(attention_cell, cell_list)#11

    if hparams.pass_hidden_state:
      decoder_initial_state = tuple(
        zs.clone(cell_state=es)
        if isinstance(zs, attention_wrapper.AttentionWrapperState) else es
        for zs, es in zip(
            cell.zero_state(batch_size, dtype), encoder_state))
    else:
      decoder_initial_state = cell.zero_state(batch_size, dtype)

    return cell, decoder_initial_state

  def _build_decoder_cudnn(self, encoder_outputs, encoder_state, hparams):
    pass
   
  def _build_decoder_fused_for_training(self, encoder_outputs, initial_state,
                                        decoder_emb_inp, hparams):
    assert self.mode == tf.contrib.learn.ModeKeys.TRAIN
    num_bi_layers = 1
    num_unidi_decoder_layers = self.num_decoder_layers - num_bi_layers
    assert num_unidi_decoder_layers == 3

    # The 1st LSTM layer
    if self.time_major:
      batch = tf.shape(encoder_outputs)[1]
      tgt_max_len = tf.shape(decoder_emb_inp)[0]
      # [batch_size] -> scalar
      initial_attention = tf.zeros(
          shape=[tgt_max_len, batch, hparams.num_units], dtype=self.dtype)
    else:
      batch = tf.shape(encoder_outputs)[0]
      tgt_max_len = tf.shape(decoder_emb_inp)[1]
      initial_attention = tf.zeros(
          shape=[batch, tgt_max_len, hparams.num_units], dtype=self.dtype)

    # Concat with initial attention
    dec_inp = tf.concat([decoder_emb_inp, initial_attention], axis=-1)

    # [tgt_time, batch, units]
    # var_scope naming chosen to agree with inference graph.
    with tf.variable_scope("multi_rnn_cell/cell_0_attention/attention"):
      outputs, _ = self._build_unidi_rnn_fused(
          dec_inp,
          initial_state[0],
          self.features["target_sequence_length"],
          hparams,
          self.dtype)
    # Get attention
    # Fused attention layer has memory of shape [batch, src_time, ...]
    if self.time_major:
      memory = tf.transpose(encoder_outputs, [1, 0, 2])
    else:
      memory = encoder_outputs
    fused_attention_layer = attention_wrapper.BahdanauAttentionFusedLayer(
        hparams.num_units, memory,
        memory_sequence_length=self.features["source_sequence_length"],
        dtype=self.dtype)
    # [batch, tgt_time, units]
    if self.time_major:
      queries = tf.transpose(outputs, [1, 0, 2])
    else:
      queries = outputs
    fused_attention = fused_attention_layer(queries)

    if self.time_major:
      # [tgt_time, batch, units]
      fused_attention = tf.transpose(fused_attention, [1, 0, 2])

    # 2-4th layer
    inputs = outputs
    for i in range(num_unidi_decoder_layers):
      # [tgt_time, batch, 2 * units]
      concat_inputs = tf.concat([inputs, fused_attention], axis=-1)

      # var_scope naming chosen to agree with inference graph.
      with tf.variable_scope("multi_rnn_cell/cell_%d" % (i+1)):
        outputs, _ = self._build_unidi_rnn_fused(
            concat_inputs, initial_state[i + 1],
            self.features["target_sequence_length"], hparams, self.dtype)
      if i >= num_unidi_decoder_layers - self.num_decoder_residual_layers:
        # gnmt_v2 attention adds the original inputs.
        outputs += inputs
      inputs = outputs
    return outputs

#11
class GNMTAttentionMultiCell(tf.nn.rnn_cell.MultiRNNCell):
  """A MultiCell with GNMT attention style."""

  def __init__(self, attention_cell, cells):
    """Creates a GNMTAttentionMultiCell.

    Args:
      attention_cell: An instance of AttentionWrapper.
      cells: A list of RNNCell wrapped with AttentionInputWrapper.
    """
    cells = [attention_cell] + cells
    super(GNMTAttentionMultiCell, self).__init__(cells, state_is_tuple=True)
  #11
  def __call__(self, inputs, state, scope=None):
    """Run the cell with bottom layer's attention copied to all upper layers."""
    if not tf.contrib.framework.nest.is_sequence(state):
      raise ValueError(
          "Expected state to be a tuple of length %d, but received: %s"
          % (len(self.state_size), state))

    with tf.variable_scope(scope or "multi_rnn_cell"):
      new_states = []

      with tf.variable_scope("cell_0_attention"):
        attention_cell = self._cells[0]
        attention_state = state[0]
        cur_inp, new_attention_state = attention_cell(inputs, attention_state)
        new_states.append(new_attention_state)

      for i in range(1, len(self._cells)):
        with tf.variable_scope("cell_%d" % i):
          cell = self._cells[i]
          cur_state = state[i]

          cur_inp = tf.concat([cur_inp, new_attention_state.attention], -1)
          cur_inp, new_state = cell(cur_inp, cur_state)
          new_states.append(new_state)

    return cur_inp, tuple(new_states)

#5
def gnmt_residual_fn(inputs, outputs):
  def split_input(inp, out):
    inp_dim = inp.get_shape().as_list()[-1]
    out_dim = out.get_shape().as_list()[-1]
    return tf.split(inp, [out_dim, inp_dim - out_dim], axis=-1)
  actual_inputs, _ = tf.contrib.framework.nest.map_structure(
      split_input, inputs, outputs)
  def assert_shape_match(inp, out):
    inp.get_shape().assert_is_compatible_with(out.get_shape())
  tf.contrib.framework.nest.assert_same_structure(actual_inputs, outputs)
  tf.contrib.framework.nest.map_structure(
      assert_shape_match, actual_inputs, outputs)
  return tf.contrib.framework.nest.map_structure(
      lambda inp, out: inp + out, actual_inputs, outputs)
