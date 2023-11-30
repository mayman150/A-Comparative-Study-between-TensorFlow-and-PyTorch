description: Turns positive integers (indexes) into dense vectors of fixed size.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.keras.layers.Embedding" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
</div>

# tf.keras.layers.Embedding

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/layers/core/embedding.py#L33-L306">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Turns positive integers (indexes) into dense vectors of fixed size.

Inherits From: [`Layer`](../../../tf/keras/layers/Layer.md), [`Module`](../../../tf/Module.md)

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.keras.layers.Embedding(
    input_dim,
    output_dim,
    embeddings_initializer=&#x27;uniform&#x27;,
    embeddings_regularizer=None,
    activity_regularizer=None,
    embeddings_constraint=None,
    mask_zero=False,
    input_length=None,
    sparse=False,
    **kwargs
)
</code></pre>



<!-- Placeholder for "Used in" -->

e.g. `[[4], [20]] -> [[0.25, 0.1], [0.6, -0.2]]`

This layer can only be used on positive integer inputs of a fixed range. The
<a href="../../../tf/keras/layers/TextVectorization.md"><code>tf.keras.layers.TextVectorization</code></a>, <a href="../../../tf/keras/layers/StringLookup.md"><code>tf.keras.layers.StringLookup</code></a>,
and <a href="../../../tf/keras/layers/IntegerLookup.md"><code>tf.keras.layers.IntegerLookup</code></a> preprocessing layers can help prepare
inputs for an `Embedding` layer.

This layer accepts <a href="../../../tf/Tensor.md"><code>tf.Tensor</code></a>, <a href="../../../tf/RaggedTensor.md"><code>tf.RaggedTensor</code></a> and <a href="../../../tf/sparse/SparseTensor.md"><code>tf.SparseTensor</code></a>
input.

#### Example:



```
>>> model = tf.keras.Sequential()
>>> model.add(tf.keras.layers.Embedding(1000, 64, input_length=10))
>>> # The model will take as input an integer matrix of size (batch,
>>> # input_length), and the largest integer (i.e. word index) in the input
>>> # should be no larger than 999 (vocabulary size).
>>> # Now model.output_shape is (None, 10, 64), where `None` is the batch
>>> # dimension.
>>> input_array = np.random.randint(1000, size=(32, 10))
>>> model.compile('rmsprop', 'mse')
>>> output_array = model.predict(input_array)
>>> print(output_array.shape)
(32, 10, 64)
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`input_dim`<a id="input_dim"></a>
</td>
<td>
Integer. Size of the vocabulary,
i.e. maximum integer index + 1.
</td>
</tr><tr>
<td>
`output_dim`<a id="output_dim"></a>
</td>
<td>
Integer. Dimension of the dense embedding.
</td>
</tr><tr>
<td>
`embeddings_initializer`<a id="embeddings_initializer"></a>
</td>
<td>
Initializer for the `embeddings`
matrix (see <a href="../../../tf/keras/initializers.md"><code>keras.initializers</code></a>).
</td>
</tr><tr>
<td>
`embeddings_regularizer`<a id="embeddings_regularizer"></a>
</td>
<td>
Regularizer function applied to
the `embeddings` matrix (see <a href="../../../tf/keras/regularizers.md"><code>keras.regularizers</code></a>).
</td>
</tr><tr>
<td>
`embeddings_constraint`<a id="embeddings_constraint"></a>
</td>
<td>
Constraint function applied to
the `embeddings` matrix (see <a href="../../../tf/keras/constraints.md"><code>keras.constraints</code></a>).
</td>
</tr><tr>
<td>
`mask_zero`<a id="mask_zero"></a>
</td>
<td>
Boolean, whether or not the input value 0 is a special
"padding" value that should be masked out. This is useful when using
recurrent layers which may take variable length input. If this is
`True`, then all subsequent layers in the model need to support masking
or an exception will be raised. If mask_zero is set to True, as a
consequence, index 0 cannot be used in the vocabulary (input_dim should
equal size of vocabulary + 1).
</td>
</tr><tr>
<td>
`input_length`<a id="input_length"></a>
</td>
<td>
Length of input sequences, when it is constant.
This argument is required if you are going to connect
`Flatten` then `Dense` layers upstream
(without it, the shape of the dense outputs cannot be computed).
</td>
</tr><tr>
<td>
`sparse`<a id="sparse"></a>
</td>
<td>
If True, calling this layer returns a <a href="../../../tf/sparse/SparseTensor.md"><code>tf.SparseTensor</code></a>. If False,
the layer returns a dense <a href="../../../tf/Tensor.md"><code>tf.Tensor</code></a>. For an entry with no features in
a sparse tensor (entry with value 0), the embedding vector of index 0 is
returned by default.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Input shape</h2></th></tr>
<tr class="alt">
<td colspan="2">
2D tensor with shape: `(batch_size, input_length)`.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Output shape</h2></th></tr>
<tr class="alt">
<td colspan="2">
3D tensor with shape: `(batch_size, input_length, output_dim)`.
</td>
</tr>

</table>


**Note on variable placement:**
By default, if a GPU is available, the embedding matrix will be placed on
the GPU. This achieves the best performance, but it might cause issues:

- You may be using an optimizer that does not support sparse GPU kernels.
In this case you will see an error upon training your model.
- Your embedding matrix may be too large to fit on your GPU. In this case
you will see an Out Of Memory (OOM) error.

In such cases, you should place the embedding matrix on the CPU memory.
You can do so with a device scope, as such:

```python
with tf.device('cpu:0'):
  embedding_layer = Embedding(...)
  embedding_layer.build()
```

The pre-built `embedding_layer` instance can then be added to a `Sequential`
model (e.g. `model.add(embedding_layer)`), called in a Functional model
(e.g. `x = embedding_layer(x)`), or used in a subclassed model.

