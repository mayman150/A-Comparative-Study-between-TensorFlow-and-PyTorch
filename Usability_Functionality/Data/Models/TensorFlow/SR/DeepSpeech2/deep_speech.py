from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
# pylint: disable=g-bad-import-order
from absl import app as absl_app
from absl import flags
from absl import logging
import tensorflow as tf
# pylint: enable=g-bad-import-order

import data.dataset as dataset
import decoder
import deep_speech_model
from official.utils.flags import core as flags_core
from official.utils.misc import distribution_utils
from official.utils.misc import model_helpers

# Default vocabulary file
_VOCABULARY_FILE = os.path.join(
    os.path.dirname(__file__), "data/vocabulary.txt")
# Evaluation metrics
_WER_KEY = "WER"
_CER_KEY = "CER"
#total = 57+8 = 65
#8
def compute_length_after_conv(max_time_steps, ctc_time_steps, input_length):
  ctc_input_length = tf.cast(tf.multiply(
      input_length, ctc_time_steps), dtype=tf.float32)
  return tf.cast(tf.math.floordiv(
      ctc_input_length, tf.cast(max_time_steps, dtype=tf.float32)), dtype=tf.int32)

#3
def evaluate_model(estimator, speech_labels, entries, input_fn_eval):

  # Get predictions
  predictions = estimator.predict(input_fn=input_fn_eval)#1

  # Get probabilities of each predicted class
  probs = [pred["probabilities"] for pred in predictions]

  num_of_examples = len(probs)
  targets = [entry[2] for entry in entries]  # The ground truth transcript

  total_wer, total_cer = 0, 0
  greedy_decoder = decoder.DeepSpeechDecoder(speech_labels)#0
  for i in range(num_of_examples):
    # Decode string.
    decoded_str = greedy_decoder.decode(probs[i])
    # Compute CER.
    total_cer += greedy_decoder.cer(decoded_str, targets[i]) / float(
        len(targets[i]))
    # Compute WER.
    total_wer += greedy_decoder.wer(decoded_str, targets[i]) / float(
        len(targets[i].split()))

  # Get mean value
  total_cer /= num_of_examples
  total_wer /= num_of_examples

  global_step = estimator.get_variable_value(tf.compat.v1.GraphKeys.GLOBAL_STEP)#1
  eval_results = {
      _WER_KEY: total_wer,
      _CER_KEY: total_cer,
      tf.compat.v1.GraphKeys.GLOBAL_STEP: global_step,#1
  }

  return eval_results

#54
def model_fn(features, labels, mode, params):
  num_classes = params["num_classes"]
  input_length = features["input_length"]
  label_length = features["label_length"]
  features = features["features"]

  # Create DeepSpeech2 model.
  model = deep_speech_model.DeepSpeech2(#17
      flags_obj.rnn_hidden_layers, flags_obj.rnn_type,
      flags_obj.is_bidirectional, flags_obj.rnn_hidden_size,
      num_classes, flags_obj.use_bias)

  if mode == tf.estimator.ModeKeys.PREDICT:#1
    logits = model(features, training=False)
    predictions = {#1
        "classes": tf.argmax(logits, axis=2),
        "probabilities": logits,
        "logits": logits
    }
    return tf.estimator.EstimatorSpec(#1
        mode=mode,
        predictions=predictions)

  # In training mode.
  logits = model(features, training=True)#17
  ctc_input_length = compute_length_after_conv(#3
      tf.shape(features)[1], tf.shape(logits)[1], input_length)
  # Compute CTC loss
  loss = tf.reduce_mean(tf.keras.backend.ctc_batch_cost(#3
      labels, logits, ctc_input_length, label_length))

  optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=flags_obj.learning_rate)#1
  global_step = tf.compat.v1.train.get_or_create_global_step()#1
  minimize_op = optimizer.minimize(loss, global_step=global_step)#1
  update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)#1
  # Create the train_op that groups both minimize_ops and update_ops
  train_op = tf.group(minimize_op, update_ops)#1

  return tf.estimator.EstimatorSpec(#1
      mode=mode,
      loss=loss,
      train_op=train_op)