description: Dot-product attention layer, a.k.a. Luong-style attention.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.keras.layers.Attention" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
</div>

# tf.keras.layers.Attention

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/layers/attention/attention.py#L30-L204">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Dot-product attention layer, a.k.a. Luong-style attention.

Inherits From: [`Layer`](../../../tf/keras/layers/Layer.md), [`Module`](../../../tf/Module.md)

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.keras.layers.Attention(
    use_scale=False, score_mode=&#x27;dot&#x27;, **kwargs
)
</code></pre>



<!-- Placeholder for "Used in" -->

Inputs are `query` tensor of shape `[batch_size, Tq, dim]`, `value` tensor
of shape `[batch_size, Tv, dim]` and `key` tensor of shape
`[batch_size, Tv, dim]`. The calculation follows the steps:

1. Calculate scores with shape `[batch_size, Tq, Tv]` as a `query`-`key` dot
    product: `scores = tf.matmul(query, key, transpose_b=True)`.
2. Use scores to calculate a distribution with shape
    `[batch_size, Tq, Tv]`: `distribution = tf.nn.softmax(scores)`.
3. Use `distribution` to create a linear combination of `value` with
     shape `[batch_size, Tq, dim]`:
     `return tf.matmul(distribution, value)`.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`use_scale`<a id="use_scale"></a>
</td>
<td>
If `True`, will create a scalar variable to scale the
attention scores.
</td>
</tr><tr>
<td>
`dropout`<a id="dropout"></a>
</td>
<td>
Float between 0 and 1. Fraction of the units to drop for the
attention scores. Defaults to 0.0.
</td>
</tr><tr>
<td>
`score_mode`<a id="score_mode"></a>
</td>
<td>
Function to use to compute attention scores, one of
`{"dot", "concat"}`. `"dot"` refers to the dot product between the
query and key vectors. `"concat"` refers to the hyperbolic tangent
of the concatenation of the query and key vectors.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Call arguments</h2></th></tr>

<tr>
<td>
`inputs`<a id="inputs"></a>
</td>
<td>
List of the following tensors:
* query: Query `Tensor` of shape `[batch_size, Tq, dim]`.
* value: Value `Tensor` of shape `[batch_size, Tv, dim]`.
* key: Optional key `Tensor` of shape `[batch_size, Tv, dim]`. If
    not given, will use `value` for both `key` and `value`, which is
    the most common case.
</td>
</tr><tr>
<td>
`mask`<a id="mask"></a>
</td>
<td>
List of the following tensors:
* query_mask: A boolean mask `Tensor` of shape `[batch_size, Tq]`.
    If given, the output will be zero at the positions where
    `mask==False`.
* value_mask: A boolean mask `Tensor` of shape `[batch_size, Tv]`.
    If given, will apply the mask such that values at positions
     where `mask==False` do not contribute to the result.
</td>
</tr><tr>
<td>
`return_attention_scores`<a id="return_attention_scores"></a>
</td>
<td>
bool, it `True`, returns the attention scores
(after masking and softmax) as an additional output argument.
</td>
</tr><tr>
<td>
`training`<a id="training"></a>
</td>
<td>
Python boolean indicating whether the layer should behave in
training mode (adding dropout) or in inference mode (no dropout).
</td>
</tr><tr>
<td>
`use_causal_mask`<a id="use_causal_mask"></a>
</td>
<td>
Boolean. Set to `True` for decoder self-attention. Adds
a mask such that position `i` cannot attend to positions `j > i`.
This prevents the flow of information from the future towards the
past.
Defaults to `False`.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Output</h2></th></tr>
<tr class="alt">
<td colspan="2">
Attention outputs of shape `[batch_size, Tq, dim]`.
[Optional] Attention scores after masking and softmax with shape
    `[batch_size, Tq, Tv]`.
</td>
</tr>

</table>


The meaning of `query`, `value` and `key` depend on the application. In the
case of text similarity, for example, `query` is the sequence embeddings of
the first piece of text and `value` is the sequence embeddings of the second
piece of text. `key` is usually the same tensor as `value`.

Here is a code example for using `Attention` in a CNN+Attention network:

```python
# Variable-length int sequences.
query_input = tf.keras.Input(shape=(None,), dtype='int32')
value_input = tf.keras.Input(shape=(None,), dtype='int32')

# Embedding lookup.
token_embedding = tf.keras.layers.Embedding(input_dim=1000, output_dim=64)
# Query embeddings of shape [batch_size, Tq, dimension].
query_embeddings = token_embedding(query_input)
# Value embeddings of shape [batch_size, Tv, dimension].
value_embeddings = token_embedding(value_input)

# CNN layer.
cnn_layer = tf.keras.layers.Conv1D(
    filters=100,
    kernel_size=4,
    # Use 'same' padding so outputs have the same shape as inputs.
    padding='same')
# Query encoding of shape [batch_size, Tq, filters].
query_seq_encoding = cnn_layer(query_embeddings)
# Value encoding of shape [batch_size, Tv, filters].
value_seq_encoding = cnn_layer(value_embeddings)

# Query-value attention of shape [batch_size, Tq, filters].
query_value_attention_seq = tf.keras.layers.Attention()(
    [query_seq_encoding, value_seq_encoding])

# Reduce over the sequence axis to produce encodings of shape
# [batch_size, filters].
query_encoding = tf.keras.layers.GlobalAveragePooling1D()(
    query_seq_encoding)
query_value_attention = tf.keras.layers.GlobalAveragePooling1D()(
    query_value_attention_seq)

# Concatenate query and document encodings to produce a DNN input layer.
input_layer = tf.keras.layers.Concatenate()(
    [query_encoding, query_value_attention])

# Add DNN layers, and create Model.
# ...
```

