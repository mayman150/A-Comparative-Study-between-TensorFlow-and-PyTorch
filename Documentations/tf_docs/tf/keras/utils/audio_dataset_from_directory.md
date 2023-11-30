description: Generates a <a href="../../../tf/data/Dataset.md"><code>tf.data.Dataset</code></a> from audio files in a directory.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.keras.utils.audio_dataset_from_directory" />
<meta itemprop="path" content="Stable" />
</div>

# tf.keras.utils.audio_dataset_from_directory

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/utils/audio_dataset.py#L31-L271">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Generates a <a href="../../../tf/data/Dataset.md"><code>tf.data.Dataset</code></a> from audio files in a directory.


<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.keras.utils.audio_dataset_from_directory(
    directory,
    labels=&#x27;inferred&#x27;,
    label_mode=&#x27;int&#x27;,
    class_names=None,
    batch_size=32,
    sampling_rate=None,
    output_sequence_length=None,
    ragged=False,
    shuffle=True,
    seed=None,
    validation_split=None,
    subset=None,
    follow_links=False
)
</code></pre>



<!-- Placeholder for "Used in" -->

If your directory structure is:

```
main_directory/
...class_a/
......a_audio_1.wav
......a_audio_2.wav
...class_b/
......b_audio_1.wav
......b_audio_2.wav
```

Then calling `audio_dataset_from_directory(main_directory,
labels='inferred')`
will return a <a href="../../../tf/data/Dataset.md"><code>tf.data.Dataset</code></a> that yields batches of audio files from
the subdirectories `class_a` and `class_b`, together with labels
0 and 1 (0 corresponding to `class_a` and 1 corresponding to `class_b`).

Only `.wav` files are supported at this time.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`directory`<a id="directory"></a>
</td>
<td>
Directory where the data is located.
If `labels` is `"inferred"`, it should contain subdirectories,
each containing audio files for a class. Otherwise, the directory
structure is ignored.
</td>
</tr><tr>
<td>
`labels`<a id="labels"></a>
</td>
<td>
Either "inferred" (labels are generated from the directory
structure), `None` (no labels), or a list/tuple of integer labels
of the same size as the number of audio files found in
the directory. Labels should be sorted according to the
alphanumeric order of the audio file paths
(obtained via `os.walk(directory)` in Python).
</td>
</tr><tr>
<td>
`label_mode`<a id="label_mode"></a>
</td>
<td>
String describing the encoding of `labels`. Options are:
- `"int"`: means that the labels are encoded as integers (e.g. for
  `sparse_categorical_crossentropy` loss).
- `"categorical"` means that the labels are encoded as a categorical
  vector (e.g. for `categorical_crossentropy` loss)
- `"binary"` means that the labels (there can be only 2)
  are encoded as `float32` scalars with values 0
  or 1 (e.g. for `binary_crossentropy`).
- `None` (no labels).
</td>
</tr><tr>
<td>
`class_names`<a id="class_names"></a>
</td>
<td>
Only valid if "labels" is `"inferred"`.
This is the explicit list of class names
(must match names of subdirectories). Used to control the order
of the classes (otherwise alphanumerical order is used).
</td>
</tr><tr>
<td>
`batch_size`<a id="batch_size"></a>
</td>
<td>
Size of the batches of data. Default: 32. If `None`,
the data will not be batched
(the dataset will yield individual samples).
</td>
</tr><tr>
<td>
`sampling_rate`<a id="sampling_rate"></a>
</td>
<td>
Audio sampling rate (in samples per second).
</td>
</tr><tr>
<td>
`output_sequence_length`<a id="output_sequence_length"></a>
</td>
<td>
Maximum length of an audio sequence. Audio files
longer than this will be truncated to `output_sequence_length`.
If set to `None`, then all sequences in the same batch will
be padded to the
length of the longest sequence in the batch.
</td>
</tr><tr>
<td>
`ragged`<a id="ragged"></a>
</td>
<td>
Whether to return a Ragged dataset (where each sequence has its
own length). Defaults to `False`.
</td>
</tr><tr>
<td>
`shuffle`<a id="shuffle"></a>
</td>
<td>
Whether to shuffle the data. Defaults to `True`.
If set to `False`, sorts the data in alphanumeric order.
</td>
</tr><tr>
<td>
`seed`<a id="seed"></a>
</td>
<td>
Optional random seed for shuffling and transformations.
</td>
</tr><tr>
<td>
`validation_split`<a id="validation_split"></a>
</td>
<td>
Optional float between 0 and 1, fraction of data to
reserve for validation.
</td>
</tr><tr>
<td>
`subset`<a id="subset"></a>
</td>
<td>
Subset of the data to return. One of `"training"`,
`"validation"` or `"both"`. Only used if `validation_split` is set.
</td>
</tr><tr>
<td>
`follow_links`<a id="follow_links"></a>
</td>
<td>
Whether to visits subdirectories pointed to by symlinks.
Defaults to `False`.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>


</table>


A <a href="../../../tf/data/Dataset.md"><code>tf.data.Dataset</code></a> object.

- If `label_mode` is `None`, it yields `string` tensors of shape
  `(batch_size,)`, containing the contents of a batch of audio files.
- Otherwise, it yields a tuple `(audio, labels)`, where `audio`
  has shape `(batch_size, sequence_length, num_channels)` and `labels`
  follows the format described
  below.

Rules regarding labels format:

- if `label_mode` is `int`, the labels are an `int32` tensor of shape
  `(batch_size,)`.
- if `label_mode` is `binary`, the labels are a `float32` tensor of
  1s and 0s of shape `(batch_size, 1)`.
- if `label_mode` is `categorical`, the labels are a `float32` tensor
  of shape `(batch_size, num_classes)`, representing a one-hot
  encoding of the class index.