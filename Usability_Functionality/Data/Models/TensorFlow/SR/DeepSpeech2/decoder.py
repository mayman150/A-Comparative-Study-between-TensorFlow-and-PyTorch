from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

from nltk.metrics import distance
import numpy as np

#0
class DeepSpeechDecoder(object):
#0
  def __init__(self, labels, blank_index=28):
    self.labels = labels
    self.blank_index = blank_index
    self.int_to_char = dict([(i, c) for (i, c) in enumerate(labels)])
#0
  def convert_to_string(self, sequence):
    return ''.join([self.int_to_char[i] for i in sequence])
#0
  def wer(self, decode, target):
    # Map each word to a new char.
    words = set(decode.split() + target.split())
    word2char = dict(zip(words, range(len(words))))

    new_decode = [chr(word2char[w]) for w in decode.split()]
    new_target = [chr(word2char[w]) for w in target.split()]

    return distance.edit_distance(''.join(new_decode), ''.join(new_target))

  def cer(self, decode, target):
    return distance.edit_distance(decode, target)

  def decode(self, logits):
    """Decode the best guess from logits using greedy algorithm."""
    # Choose the class with maximimum probability.
    best = list(np.argmax(logits, axis=1))
    # Merge repeated chars.
    merge = [k for k, _ in itertools.groupby(best)]
    # Remove the blank index in the decoded sequence.
    merge_remove_blank = []
    for k in merge:
      if k != self.blank_index:
        merge_remove_blank.append(k)

    return self.convert_to_string(merge_remove_blank)
