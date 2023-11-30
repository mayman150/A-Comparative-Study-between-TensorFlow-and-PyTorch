description: A preprocessing layer which hashes and bins categorical features.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.keras.layers.Hashing" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
</div>

# tf.keras.layers.Hashing

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/layers/preprocessing/hashing.py#L35-L298">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



A preprocessing layer which hashes and bins categorical features.

Inherits From: [`Layer`](../../../tf/keras/layers/Layer.md), [`Module`](../../../tf/Module.md)

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`tf.keras.layers.experimental.preprocessing.Hashing`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.keras.layers.Hashing(
    num_bins,
    mask_value=None,
    salt=None,
    output_mode=&#x27;int&#x27;,
    sparse=False,
    **kwargs
)
</code></pre>



<!-- Placeholder for "Used in" -->

This layer transforms categorical inputs to hashed output. It element-wise
converts a ints or strings to ints in a fixed range. The stable hash
function uses `tensorflow::ops::Fingerprint` to produce the same output
consistently across all platforms.

This layer uses [FarmHash64](https://github.com/google/farmhash) by default,
which provides a consistent hashed output across different platforms and is
stable across invocations, regardless of device and context, by mixing the
input bits thoroughly.

If you want to obfuscate the hashed output, you can also pass a random
`salt` argument in the constructor. In that case, the layer will use the
[SipHash64](https://github.com/google/highwayhash) hash function, with
the `salt` value serving as additional input to the hash function.

For an overview and full list of preprocessing layers, see the preprocessing
[guide](https://www.tensorflow.org/guide/keras/preprocessing_layers).

**Example (FarmHash64)**

```
>>> layer = tf.keras.layers.Hashing(num_bins=3)
>>> inp = [['A'], ['B'], ['C'], ['D'], ['E']]
>>> layer(inp)
<tf.Tensor: shape=(5, 1), dtype=int64, numpy=
  array([[1],
         [0],
         [1],
         [1],
         [2]])>
```

**Example (FarmHash64) with a mask value**

```
>>> layer = tf.keras.layers.Hashing(num_bins=3, mask_value='')
>>> inp = [['A'], ['B'], [''], ['C'], ['D']]
>>> layer(inp)
<tf.Tensor: shape=(5, 1), dtype=int64, numpy=
  array([[1],
         [1],
         [0],
         [2],
         [2]])>
```

**Example (SipHash64)**

```
>>> layer = tf.keras.layers.Hashing(num_bins=3, salt=[133, 137])
>>> inp = [['A'], ['B'], ['C'], ['D'], ['E']]
>>> layer(inp)
<tf.Tensor: shape=(5, 1), dtype=int64, numpy=
  array([[1],
         [2],
         [1],
         [0],
         [2]])>
```

**Example (Siphash64 with a single integer, same as `salt=[133, 133]`)**

```
>>> layer = tf.keras.layers.Hashing(num_bins=3, salt=133)
>>> inp = [['A'], ['B'], ['C'], ['D'], ['E']]
>>> layer(inp)
<tf.Tensor: shape=(5, 1), dtype=int64, numpy=
  array([[0],
         [0],
         [2],
         [1],
         [0]])>
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`num_bins`<a id="num_bins"></a>
</td>
<td>
Number of hash bins. Note that this includes the `mask_value`
bin, so the effective number of bins is `(num_bins - 1)` if `mask_value`
is set.
</td>
</tr><tr>
<td>
`mask_value`<a id="mask_value"></a>
</td>
<td>
A value that represents masked inputs, which are mapped to
index 0. `None` means no mask term will be added and the
hashing will start at index 0. Defaults to `None`.
</td>
</tr><tr>
<td>
`salt`<a id="salt"></a>
</td>
<td>
A single unsigned integer or None.
If passed, the hash function used will be SipHash64, with these values
used as an additional input (known as a "salt" in cryptography).
These should be non-zero. If `None`, uses the FarmHash64 hash function.
It also supports tuple/list of 2 unsigned integer numbers, see
reference paper for details. Defaults to `None`.
</td>
</tr><tr>
<td>
`output_mode`<a id="output_mode"></a>
</td>
<td>
Specification for the output of the layer. Values can bes
`"int"`, `"one_hot"`, `"multi_hot"`, or
`"count"` configuring the layer as follows:
  - `"int"`: Return the integer bin indices directly.
  - `"one_hot"`: Encodes each individual element in the input into an
    array the same size as `num_bins`, containing a 1 at the input's bin
    index. If the last dimension is size 1, will encode on that
    dimension.  If the last dimension is not size 1, will append a new
    dimension for the encoded output.
  - `"multi_hot"`: Encodes each sample in the input into a single array
    the same size as `num_bins`, containing a 1 for each bin index
    index present in the sample. Treats the last dimension as the sample
    dimension, if input shape is `(..., sample_length)`, output shape
    will be `(..., num_tokens)`.
  - `"count"`: As `"multi_hot"`, but the int array contains a count of
    the number of times the bin index appeared in the sample.
Defaults to `"int"`.
</td>
</tr><tr>
<td>
`sparse`<a id="sparse"></a>
</td>
<td>
Boolean. Only applicable to `"one_hot"`, `"multi_hot"`,
and `"count"` output modes. If True, returns a `SparseTensor` instead of
a dense `Tensor`. Defaults to `False`.
</td>
</tr><tr>
<td>
`**kwargs`<a id="**kwargs"></a>
</td>
<td>
Keyword arguments to construct a layer.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Input shape</h2></th></tr>
<tr class="alt">
<td colspan="2">
A single or list of string, int32 or int64 `Tensor`,
`SparseTensor` or `RaggedTensor` of shape `(batch_size, ...,)`
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Output shape</h2></th></tr>
<tr class="alt">
<td colspan="2">
An int64 `Tensor`, `SparseTensor` or `RaggedTensor` of shape
`(batch_size, ...)`. If any input is `RaggedTensor` then output is
`RaggedTensor`, otherwise if any input is `SparseTensor` then output is
`SparseTensor`, otherwise the output is `Tensor`.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Reference</h2></th></tr>
<tr class="alt">
<td colspan="2">
- [SipHash with salt](https://www.131002.net/siphash/siphash.pdf)
</td>
</tr>

</table>



