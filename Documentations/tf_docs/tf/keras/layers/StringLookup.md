description: A preprocessing layer which maps string features to integer indices.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.keras.layers.StringLookup" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="adapt"/>
<meta itemprop="property" content="compile"/>
<meta itemprop="property" content="get_vocabulary"/>
<meta itemprop="property" content="load_assets"/>
<meta itemprop="property" content="reset_state"/>
<meta itemprop="property" content="save_assets"/>
<meta itemprop="property" content="set_vocabulary"/>
<meta itemprop="property" content="update_state"/>
<meta itemprop="property" content="vocab_size"/>
<meta itemprop="property" content="vocabulary_size"/>
</div>

# tf.keras.layers.StringLookup

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/layers/preprocessing/string_lookup.py#L27-L417">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



A preprocessing layer which maps string features to integer indices.

Inherits From: [`PreprocessingLayer`](../../../tf/keras/layers/experimental/preprocessing/PreprocessingLayer.md), [`Layer`](../../../tf/keras/layers/Layer.md), [`Module`](../../../tf/Module.md)

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`tf.keras.layers.experimental.preprocessing.StringLookup`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.keras.layers.StringLookup(
    max_tokens=None,
    num_oov_indices=1,
    mask_token=None,
    oov_token=&#x27;[UNK]&#x27;,
    vocabulary=None,
    idf_weights=None,
    encoding=&#x27;utf-8&#x27;,
    invert=False,
    output_mode=&#x27;int&#x27;,
    sparse=False,
    pad_to_max_tokens=False,
    **kwargs
)
</code></pre>



<!-- Placeholder for "Used in" -->

This layer translates a set of arbitrary strings into integer output via a
table-based vocabulary lookup. This layer will perform no splitting or
transformation of input strings. For a layer that can split and tokenize
natural language, see the <a href="../../../tf/keras/layers/TextVectorization.md"><code>tf.keras.layers.TextVectorization</code></a> layer.

The vocabulary for the layer must be either supplied on construction or
learned via `adapt()`. During `adapt()`, the layer will analyze a data set,
determine the frequency of individual strings tokens, and create a
vocabulary from them. If the vocabulary is capped in size, the most frequent
tokens will be used to create the vocabulary and all others will be treated
as out-of-vocabulary (OOV).

There are two possible output modes for the layer.
When `output_mode` is `"int"`,
input strings are converted to their index in the vocabulary (an integer).
When `output_mode` is `"multi_hot"`, `"count"`, or `"tf_idf"`, input strings
are encoded into an array where each dimension corresponds to an element in
the vocabulary.

The vocabulary can optionally contain a mask token as well as an OOV token
(which can optionally occupy multiple indices in the vocabulary, as set
by `num_oov_indices`).
The position of these tokens in the vocabulary is fixed. When `output_mode`
is `"int"`, the vocabulary will begin with the mask token (if set), followed
by OOV indices, followed by the rest of the vocabulary. When `output_mode`
is `"multi_hot"`, `"count"`, or `"tf_idf"` the vocabulary will begin with
OOV indices and instances of the mask token will be dropped.

For an overview and full list of preprocessing layers, see the preprocessing
[guide](https://www.tensorflow.org/guide/keras/preprocessing_layers).

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`max_tokens`<a id="max_tokens"></a>
</td>
<td>
Maximum size of the vocabulary for this layer. This should
only be specified when adapting the vocabulary or when setting
`pad_to_max_tokens=True`. If None, there is no cap on the size of the
vocabulary. Note that this size includes the OOV and mask tokens.
Defaults to `None`.
</td>
</tr><tr>
<td>
`num_oov_indices`<a id="num_oov_indices"></a>
</td>
<td>
The number of out-of-vocabulary tokens to use. If this
value is more than 1, OOV inputs are hashed to determine their OOV
value. If this value is 0, OOV inputs will cause an error when calling
the layer.  Defaults to `1`.
</td>
</tr><tr>
<td>
`mask_token`<a id="mask_token"></a>
</td>
<td>
A token that represents masked inputs. When `output_mode` is
`"int"`, the token is included in vocabulary and mapped to index 0. In
other output modes, the token will not appear in the vocabulary and
instances of the mask token in the input will be dropped. If set to
None, no mask term will be added. Defaults to `None`.
</td>
</tr><tr>
<td>
`oov_token`<a id="oov_token"></a>
</td>
<td>
Only used when `invert` is True. The token to return for OOV
indices. Defaults to `"[UNK]"`.
</td>
</tr><tr>
<td>
`vocabulary`<a id="vocabulary"></a>
</td>
<td>
Optional. Either an array of strings or a string path to a
text file. If passing an array, can pass a tuple, list, 1D numpy array,
or 1D tensor containing the string vocabulary terms. If passing a file
path, the file should contain one line per term in the vocabulary. If
this argument is set, there is no need to `adapt()` the layer.
</td>
</tr><tr>
<td>
`idf_weights`<a id="idf_weights"></a>
</td>
<td>
Only valid when `output_mode` is `"tf_idf"`. A tuple, list,
1D numpy array, or 1D tensor or the same length as the vocabulary,
containing the floating point inverse document frequency weights, which
will be multiplied by per sample term counts for the final `tf_idf`
weight. If the `vocabulary` argument is set, and `output_mode` is
`"tf_idf"`, this argument must be supplied.
</td>
</tr><tr>
<td>
`invert`<a id="invert"></a>
</td>
<td>
Only valid when `output_mode` is `"int"`. If True, this layer will
map indices to vocabulary items instead of mapping vocabulary items to
indices. Defaults to `False`.
</td>
</tr><tr>
<td>
`output_mode`<a id="output_mode"></a>
</td>
<td>
Specification for the output of the layer. Values can be
`"int"`, `"one_hot"`, `"multi_hot"`, `"count"`, or `"tf_idf"`
configuring the layer as follows:
  - `"int"`: Return the raw integer indices of the input tokens.
  - `"one_hot"`: Encodes each individual element in the input into an
    array the same size as the vocabulary, containing a 1 at the element
    index. If the last dimension is size 1, will encode on that
    dimension. If the last dimension is not size 1, will append a new
    dimension for the encoded output.
  - `"multi_hot"`: Encodes each sample in the input into a single array
    the same size as the vocabulary, containing a 1 for each vocabulary
    term present in the sample. Treats the last dimension as the sample
    dimension, if input shape is (..., sample_length), output shape will
    be (..., num_tokens).
  - `"count"`: As `"multi_hot"`, but the int array contains a count of
    the number of times the token at that index appeared in the sample.
  - `"tf_idf"`: As `"multi_hot"`, but the TF-IDF algorithm is applied to
    find the value in each token slot.
For `"int"` output, any shape of input and output is supported. For all
other output modes, currently only output up to rank 2 is supported.
Defaults to `"int"`.
</td>
</tr><tr>
<td>
`pad_to_max_tokens`<a id="pad_to_max_tokens"></a>
</td>
<td>
Only applicable when `output_mode` is `"multi_hot"`,
`"count"`, or `"tf_idf"`. If True, the output will have its feature axis
padded to `max_tokens` even if the number of unique tokens in the
vocabulary is less than max_tokens, resulting in a tensor of shape
[batch_size, max_tokens] regardless of vocabulary size. Defaults to
`False`.
</td>
</tr><tr>
<td>
`sparse`<a id="sparse"></a>
</td>
<td>
Boolean. Only applicable when `output_mode` is `"multi_hot"`,
`"count"`, or `"tf_idf"`. If True, returns a `SparseTensor` instead of a
dense `Tensor`. Defaults to `False`.
</td>
</tr><tr>
<td>
`encoding`<a id="encoding"></a>
</td>
<td>
Optional. The text encoding to use to interpret the input
strings. Defaults to `"utf-8"`.
</td>
</tr>
</table>



#### Examples:



**Creating a lookup layer with a known vocabulary**

This example creates a lookup layer with a pre-existing vocabulary.

```
>>> vocab = ["a", "b", "c", "d"]
>>> data = tf.constant([["a", "c", "d"], ["d", "z", "b"]])
>>> layer = tf.keras.layers.StringLookup(vocabulary=vocab)
>>> layer(data)
<tf.Tensor: shape=(2, 3), dtype=int64, numpy=
array([[1, 3, 4],
       [4, 0, 2]])>
```

**Creating a lookup layer with an adapted vocabulary**

This example creates a lookup layer and generates the vocabulary by
analyzing the dataset.

```
>>> data = tf.constant([["a", "c", "d"], ["d", "z", "b"]])
>>> layer = tf.keras.layers.StringLookup()
>>> layer.adapt(data)
>>> layer.get_vocabulary()
['[UNK]', 'd', 'z', 'c', 'b', 'a']
```

Note that the OOV token `"[UNK]"` has been added to the vocabulary.
The remaining tokens are sorted by frequency
(`"d"`, which has 2 occurrences, is first) then by inverse sort order.

```
>>> data = tf.constant([["a", "c", "d"], ["d", "z", "b"]])
>>> layer = tf.keras.layers.StringLookup()
>>> layer.adapt(data)
>>> layer(data)
<tf.Tensor: shape=(2, 3), dtype=int64, numpy=
array([[5, 3, 1],
       [1, 2, 4]])>
```

**Lookups with multiple OOV indices**

This example demonstrates how to use a lookup layer with multiple OOV
indices.  When a layer is created with more than one OOV index, any OOV
values are hashed into the number of OOV buckets, distributing OOV values in
a deterministic fashion across the set.

```
>>> vocab = ["a", "b", "c", "d"]
>>> data = tf.constant([["a", "c", "d"], ["m", "z", "b"]])
>>> layer = tf.keras.layers.StringLookup(vocabulary=vocab,
...                                      num_oov_indices=2)
>>> layer(data)
<tf.Tensor: shape=(2, 3), dtype=int64, numpy=
array([[2, 4, 5],
       [0, 1, 3]])>
```

Note that the output for OOV value 'm' is 0, while the output for OOV value
'z' is 1. The in-vocab terms have their output index increased by 1 from
earlier examples (a maps to 2, etc) in order to make space for the extra OOV
value.

**One-hot output**

Configure the layer with `output_mode='one_hot'`. Note that the first
`num_oov_indices` dimensions in the one_hot encoding represent OOV values.

```
>>> vocab = ["a", "b", "c", "d"]
>>> data = tf.constant(["a", "b", "c", "d", "z"])
>>> layer = tf.keras.layers.StringLookup(
...     vocabulary=vocab, output_mode='one_hot')
>>> layer(data)
<tf.Tensor: shape=(5, 5), dtype=float32, numpy=
  array([[0., 1., 0., 0., 0.],
         [0., 0., 1., 0., 0.],
         [0., 0., 0., 1., 0.],
         [0., 0., 0., 0., 1.],
         [1., 0., 0., 0., 0.]], dtype=float32)>
```

**Multi-hot output**

Configure the layer with `output_mode='multi_hot'`. Note that the first
`num_oov_indices` dimensions in the multi_hot encoding represent OOV values.

```
>>> vocab = ["a", "b", "c", "d"]
>>> data = tf.constant([["a", "c", "d", "d"], ["d", "z", "b", "z"]])
>>> layer = tf.keras.layers.StringLookup(
...     vocabulary=vocab, output_mode='multi_hot')
>>> layer(data)
<tf.Tensor: shape=(2, 5), dtype=float32, numpy=
  array([[0., 1., 0., 1., 1.],
         [1., 0., 1., 0., 1.]], dtype=float32)>
```

**Token count output**

Configure the layer with `output_mode='count'`. As with multi_hot output,
the first `num_oov_indices` dimensions in the output represent OOV values.

```
>>> vocab = ["a", "b", "c", "d"]
>>> data = tf.constant([["a", "c", "d", "d"], ["d", "z", "b", "z"]])
>>> layer = tf.keras.layers.StringLookup(
...     vocabulary=vocab, output_mode='count')
>>> layer(data)
<tf.Tensor: shape=(2, 5), dtype=float32, numpy=
  array([[0., 1., 0., 1., 2.],
         [2., 0., 1., 0., 1.]], dtype=float32)>
```

**TF-IDF output**

Configure the layer with `output_mode="tf_idf"`. As with multi_hot output,
the first `num_oov_indices` dimensions in the output represent OOV values.

Each token bin will output `token_count * idf_weight`, where the idf weights
are the inverse document frequency weights per token. These should be
provided along with the vocabulary. Note that the `idf_weight` for OOV
values will default to the average of all idf weights passed in.

```
>>> vocab = ["a", "b", "c", "d"]
>>> idf_weights = [0.25, 0.75, 0.6, 0.4]
>>> data = tf.constant([["a", "c", "d", "d"], ["d", "z", "b", "z"]])
>>> layer = tf.keras.layers.StringLookup(output_mode="tf_idf")
>>> layer.set_vocabulary(vocab, idf_weights=idf_weights)
>>> layer(data)
<tf.Tensor: shape=(2, 5), dtype=float32, numpy=
  array([[0.  , 0.25, 0.  , 0.6 , 0.8 ],
         [1.0 , 0.  , 0.75, 0.  , 0.4 ]], dtype=float32)>
```

To specify the idf weights for oov values, you will need to pass the entire
vocabularly including the leading oov token.

```
>>> vocab = ["[UNK]", "a", "b", "c", "d"]
>>> idf_weights = [0.9, 0.25, 0.75, 0.6, 0.4]
>>> data = tf.constant([["a", "c", "d", "d"], ["d", "z", "b", "z"]])
>>> layer = tf.keras.layers.StringLookup(output_mode="tf_idf")
>>> layer.set_vocabulary(vocab, idf_weights=idf_weights)
>>> layer(data)
<tf.Tensor: shape=(2, 5), dtype=float32, numpy=
  array([[0.  , 0.25, 0.  , 0.6 , 0.8 ],
         [1.8 , 0.  , 0.75, 0.  , 0.4 ]], dtype=float32)>
```

When adapting the layer in `"tf_idf"` mode, each input sample will be
considered a document, and IDF weight per token will be calculated as
`log(1 + num_documents / (1 + token_document_count))`.

**Inverse lookup**

This example demonstrates how to map indices to strings using this layer.
(You can also use `adapt()` with `inverse=True`, but for simplicity we'll
pass the vocab in this example.)

```
>>> vocab = ["a", "b", "c", "d"]
>>> data = tf.constant([[1, 3, 4], [4, 0, 2]])
>>> layer = tf.keras.layers.StringLookup(vocabulary=vocab, invert=True)
>>> layer(data)
<tf.Tensor: shape=(2, 3), dtype=string, numpy=
array([[b'a', b'c', b'd'],
       [b'd', b'[UNK]', b'b']], dtype=object)>
```

Note that the first index correspond to the oov token by default.


**Forward and inverse lookup pairs**

This example demonstrates how to use the vocabulary of a standard lookup
layer to create an inverse lookup layer.

```
>>> vocab = ["a", "b", "c", "d"]
>>> data = tf.constant([["a", "c", "d"], ["d", "z", "b"]])
>>> layer = tf.keras.layers.StringLookup(vocabulary=vocab)
>>> i_layer = tf.keras.layers.StringLookup(vocabulary=vocab, invert=True)
>>> int_data = layer(data)
>>> i_layer(int_data)
<tf.Tensor: shape=(2, 3), dtype=string, numpy=
array([[b'a', b'c', b'd'],
       [b'd', b'[UNK]', b'b']], dtype=object)>
```

In this example, the input value `"z"` resulted in an output of `"[UNK]"`,
since 1000 was not in the vocabulary - it got represented as an OOV, and all
OOV values are returned as `"[UNK]"` in the inverse layer. Also, note that
for the inverse to work, you must have already set the forward layer
vocabulary either directly or via `adapt()` before calling
`get_vocabulary()`.



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`is_adapted`<a id="is_adapted"></a>
</td>
<td>
Whether the layer has been fit to data already.
</td>
</tr>
</table>



## Methods

<h3 id="adapt"><code>adapt</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/layers/preprocessing/string_lookup.py#L360-L410">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>adapt(
    data, batch_size=None, steps=None
)
</code></pre>

Computes a vocabulary of string terms from tokens in a dataset.

Calling `adapt()` on a `StringLookup` layer is an alternative to passing
in a precomputed vocabulary on construction via the `vocabulary`
argument. A `StringLookup` layer should always be either adapted over a
dataset or supplied with a vocabulary.

During `adapt()`, the layer will build a vocabulary of all string tokens
seen in the dataset, sorted by occurrence count, with ties broken by
sort order of the tokens (high to low). At the end of `adapt()`, if
`max_tokens` is set, the vocabulary wil be truncated to `max_tokens`
size. For example, adapting a layer with `max_tokens=1000` will compute
the 1000 most frequent tokens occurring in the input dataset. If
`output_mode='tf-idf'`, `adapt()` will also learn the document
frequencies of each token in the input dataset.

In order to make `StringLookup` efficient in any distribution context,
the vocabulary is kept static with respect to any compiled <a href="../../../tf/Graph.md"><code>tf.Graph</code></a>s
that call the layer. As a consequence, if the layer is adapted a second
time, any models using the layer should be re-compiled. For more
information see
<a href="../../../tf/keras/layers/experimental/preprocessing/PreprocessingLayer.md#adapt"><code>tf.keras.layers.experimental.preprocessing.PreprocessingLayer.adapt</code></a>.

`adapt()` is meant only as a single machine utility to compute layer
state.  To analyze a dataset that cannot fit on a single machine, see
[Tensorflow Transform](
https://www.tensorflow.org/tfx/transform/get_started) for a
multi-machine, map-reduce solution.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Arguments</th></tr>

<tr>
<td>
`data`
</td>
<td>
The data to train on. It can be passed either as a
<a href="../../../tf/data/Dataset.md"><code>tf.data.Dataset</code></a>, or as a numpy array.
</td>
</tr><tr>
<td>
`batch_size`
</td>
<td>
Integer or `None`.
Number of samples per state update.
If unspecified, `batch_size` will default to 32.
Do not specify the `batch_size` if your data is in the
form of datasets, generators, or <a href="../../../tf/keras/utils/Sequence.md"><code>keras.utils.Sequence</code></a> instances
(since they generate batches).
</td>
</tr><tr>
<td>
`steps`
</td>
<td>
Integer or `None`.
Total number of steps (batches of samples)
When training with input tensors such as
TensorFlow data tensors, the default `None` is equal to
the number of samples in your dataset divided by
the batch size, or 1 if that cannot be determined. If x is a
<a href="../../../tf/data.md"><code>tf.data</code></a> dataset, and 'steps' is None, the epoch will run until
the input dataset is exhausted. When passing an infinitely
repeating dataset, you must specify the `steps` argument. This
argument is not supported with array inputs.
</td>
</tr>
</table>



<h3 id="compile"><code>compile</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/engine/base_preprocessing_layer.py#L139-L160">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>compile(
    run_eagerly=None, steps_per_execution=None
)
</code></pre>

Configures the layer for `adapt`.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Arguments</th></tr>

<tr>
<td>
`run_eagerly`
</td>
<td>
Bool. If `True`, this `Model`'s
logic will not be wrapped in a <a href="../../../tf/function.md"><code>tf.function</code></a>. Recommended to leave
this as `None` unless your `Model` cannot be run inside a
<a href="../../../tf/function.md"><code>tf.function</code></a>. Defaults to `False`.
</td>
</tr><tr>
<td>
`steps_per_execution`
</td>
<td>
Int. The number of batches to run
during each <a href="../../../tf/function.md"><code>tf.function</code></a> call. Running multiple batches inside a
single <a href="../../../tf/function.md"><code>tf.function</code></a> call can greatly improve performance on TPUs or
small models with a large Python overhead. Defaults to `1`.
</td>
</tr>
</table>



<h3 id="get_vocabulary"><code>get_vocabulary</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/layers/preprocessing/index_lookup.py#L370-L399">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_vocabulary(
    include_special_tokens=True
)
</code></pre>

Returns the current vocabulary of the layer.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`include_special_tokens`
</td>
<td>
If True, the returned vocabulary will include
mask and OOV tokens, and a term's index in the vocabulary will equal
the term's index when calling the layer. If False, the returned
vocabulary will not include any mask or OOV tokens.
</td>
</tr>
</table>



<h3 id="load_assets"><code>load_assets</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/layers/preprocessing/index_lookup.py#L848-L864">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>load_assets(
    dir_path
)
</code></pre>




<h3 id="reset_state"><code>reset_state</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/layers/preprocessing/index_lookup.py#L727-L736">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>reset_state()
</code></pre>

Resets the statistics of the preprocessing layer.


<h3 id="save_assets"><code>save_assets</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/layers/preprocessing/index_lookup.py#L838-L846">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>save_assets(
    dir_path
)
</code></pre>




<h3 id="set_vocabulary"><code>set_vocabulary</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/layers/preprocessing/index_lookup.py#L443-L628">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>set_vocabulary(
    vocabulary, idf_weights=None
)
</code></pre>

Sets vocabulary (and optionally document frequency) for this layer.

This method sets the vocabulary and idf weights for this layer directly,
instead of analyzing a dataset through `adapt`. It should be used
whenever the vocab (and optionally document frequency) information is
already known.  If vocabulary data is already present in the layer, this
method will replace it.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`vocabulary`
</td>
<td>
Either an array or a string path to a text file. If
passing an array, can pass a tuple, list, 1D numpy array, or 1D
tensor containing the vocbulary terms. If passing a file path, the
file should contain one line per term in the vocabulary.
</td>
</tr><tr>
<td>
`idf_weights`
</td>
<td>
A tuple, list, 1D numpy array, or 1D tensor of inverse
document frequency weights with equal length to vocabulary. Must be
set if `output_mode` is `"tf_idf"`. Should not be set otherwise.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>

<tr>
<td>
`ValueError`
</td>
<td>
If there are too many inputs, the inputs do not match, or
input data is missing.
</td>
</tr><tr>
<td>
`RuntimeError`
</td>
<td>
If the vocabulary cannot be set when this function is
called. This happens when `"multi_hot"`, `"count"`, and `"tf_idf"`
modes, if `pad_to_max_tokens` is False and the layer itself has
already been called.
</td>
</tr><tr>
<td>
`RuntimeError`
</td>
<td>
If a tensor vocabulary is passed outside of eager
execution.
</td>
</tr>
</table>



<h3 id="update_state"><code>update_state</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/layers/preprocessing/index_lookup.py#L630-L664">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>update_state(
    data
)
</code></pre>

Accumulates statistics for the preprocessing layer.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Arguments</th></tr>

<tr>
<td>
`data`
</td>
<td>
A mini-batch of inputs to the layer.
</td>
</tr>
</table>



<h3 id="vocab_size"><code>vocab_size</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/layers/preprocessing/index_lookup.py#L416-L418">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>vocab_size()
</code></pre>




<h3 id="vocabulary_size"><code>vocabulary_size</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/layers/preprocessing/index_lookup.py#L401-L414">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>vocabulary_size()
</code></pre>

Gets the current size of the layer's vocabulary.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
The integer size of the vocabulary, including optional mask and oov
indices.
</td>
</tr>

</table>





