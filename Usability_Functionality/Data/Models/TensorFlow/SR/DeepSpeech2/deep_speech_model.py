from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
#total = 3+1+4+4+17 = 29
# Supported rnn cells.
#3
SUPPORTED_RNNS = {
    "lstm": tf.keras.layers.LSTMCell,
    "rnn": tf.keras.layers.SimpleRNNCell,
    "gru": tf.keras.layers.GRUCell,
}

# Parameters for batch normalization.
_BATCH_NORM_EPSILON = 1e-5
_BATCH_NORM_DECAY = 0.997

# Filters of convolution layer
_CONV_FILTERS = 32

#1
def batch_norm(inputs, training):
  return tf.keras.layers.BatchNormalization(
      momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON)(inputs, training=training)

#4
def _conv_bn_layer(inputs, padding, filters, kernel_size, strides, layer_id,
                   training):
  inputs = tf.pad(
      inputs,
      [[0, 0], [padding[0], padding[0]], [padding[1], padding[1]], [0, 0]])
  inputs = tf.keras.layers.Conv2D(
      filters=filters, kernel_size=kernel_size, strides=strides,
      padding="valid", use_bias=False, activation=tf.nn.relu6,
      name="cnn_{}".format(layer_id))(inputs)
  return batch_norm(inputs, training)

#4
def _rnn_layer(inputs, rnn_cell, rnn_hidden_size, layer_id, is_batch_norm,
               is_bidirectional, training):
  if is_batch_norm:
    inputs = batch_norm(inputs, training)#1
  if is_bidirectional:
    rnn_outputs = tf.keras.layers.Bidirectional(
        tf.keras.layers.RNN(rnn_cell(rnn_hidden_size),
                            return_sequences=True))(inputs)
  else:
    rnn_outputs = tf.keras.layers.RNN(
        rnn_cell(rnn_hidden_size), return_sequences=True)(inputs)

  return rnn_outputs
#17
class DeepSpeech2(object):
  """Define DeepSpeech2 model."""

  def __init__(self, num_rnn_layers, rnn_type, is_bidirectional,
               rnn_hidden_size, num_classes, use_bias):
    self.num_rnn_layers = num_rnn_layers
    self.rnn_type = rnn_type
    self.is_bidirectional = is_bidirectional
    self.rnn_hidden_size = rnn_hidden_size
    self.num_classes = num_classes
    self.use_bias = use_bias
  #17
  def __call__(self, inputs, training):
    # Two cnn layers.
    inputs = _conv_bn_layer(#4
        inputs, padding=(20, 5), filters=_CONV_FILTERS, kernel_size=(41, 11),
        strides=(2, 2), layer_id=1, training=training)

    inputs = _conv_bn_layer(#4
        inputs, padding=(10, 5), filters=_CONV_FILTERS, kernel_size=(21, 11),
        strides=(2, 1), layer_id=2, training=training)

    batch_size = tf.shape(inputs)[0]#1
    feat_size = inputs.get_shape().as_list()[2]
    inputs = tf.reshape(#1
        inputs,
        [batch_size, -1, feat_size * _CONV_FILTERS])

    # RNN layers.
    rnn_cell = SUPPORTED_RNNS[self.rnn_type]#1
    for layer_counter in xrange(self.num_rnn_layers):
      # No batch normalization on the first layer.
      is_batch_norm = (layer_counter != 0)
      inputs = _rnn_layer(#4
          inputs, rnn_cell, self.rnn_hidden_size, layer_counter + 1,
          is_batch_norm, self.is_bidirectional, training)

    # FC layer with batch norm.
    inputs = batch_norm(inputs, training)#1
    logits = tf.keras.layers.Dense(#1
        self.num_classes, use_bias=self.use_bias, activation="softmax")(inputs)

    return logits

