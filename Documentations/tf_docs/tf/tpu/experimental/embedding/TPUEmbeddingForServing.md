description: The TPUEmbedding mid level API running on CPU for serving.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.tpu.experimental.embedding.TPUEmbeddingForServing" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__call__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="build"/>
<meta itemprop="property" content="embedding_lookup"/>
</div>

# tf.tpu.experimental.embedding.TPUEmbeddingForServing

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/tpu/tpu_embedding_for_serving.py">View source</a>



The TPUEmbedding mid level API running on CPU for serving.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.tpu.experimental.embedding.TPUEmbeddingForServing`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.tpu.experimental.embedding.TPUEmbeddingForServing(
    feature_config: Union[<a href="../../../../tf/tpu/experimental/embedding/FeatureConfig.md"><code>tf.tpu.experimental.embedding.FeatureConfig</code></a>, Iterable],
    optimizer: Optional[tpu_embedding_v2_utils._Optimizer],
    experimental_sparsecore_restore_info: Optional[Dict[str, Any]] = None
)
</code></pre>



<!-- Placeholder for "Used in" -->

Note: This class is intended to be used for embedding tables that are trained
on TPU and to be served on CPU. Therefore the class should be only initialized
under non-TPU strategy. Otherwise an error will be raised.

You can first train your model using the TPUEmbedding class and save the
checkpoint. Then use this class to restore the checkpoint to do serving.

First train a model and save the checkpoint.
```python
model = model_fn(...)
strategy = tf.distribute.TPUStrategy(...)
with strategy.scope():
  embedding = tf.tpu.experimental.embedding.TPUEmbedding(
      feature_config=feature_config,
      optimizer=tf.tpu.experimental.embedding.SGD(0.1))

# Your custom training code.

checkpoint = tf.train.Checkpoint(model=model, embedding=embedding)
checkpoint.save(...)

```

Then restore the checkpoint and do serving.
```python

# Restore the model on CPU.
model = model_fn(...)
embedding = tf.tpu.experimental.embedding.TPUEmbeddingForServing(
      feature_config=feature_config,
      optimizer=tf.tpu.experimental.embedding.SGD(0.1))

checkpoint = tf.train.Checkpoint(model=model, embedding=embedding)
checkpoint.restore(...)

result = embedding(...)
table = embedding.embedding_table
```

NOTE: This class can also be used to do embedding training on CPU. But it
requires the conversion between keras optimizer and embedding optimizers so
that the slot variables can stay consistent between them.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`feature_config`<a id="feature_config"></a>
</td>
<td>
A nested structure of
<a href="../../../../tf/tpu/experimental/embedding/FeatureConfig.md"><code>tf.tpu.experimental.embedding.FeatureConfig</code></a> configs.
</td>
</tr><tr>
<td>
`optimizer`<a id="optimizer"></a>
</td>
<td>
An instance of one of <a href="../../../../tf/tpu/experimental/embedding/SGD.md"><code>tf.tpu.experimental.embedding.SGD</code></a>,
<a href="../../../../tf/tpu/experimental/embedding/Adagrad.md"><code>tf.tpu.experimental.embedding.Adagrad</code></a> or
<a href="../../../../tf/tpu/experimental/embedding/Adam.md"><code>tf.tpu.experimental.embedding.Adam</code></a>. When not created under TPUStrategy
may be set to None to avoid the creation of the optimizer slot
variables, useful for optimizing memory consumption when exporting the
model for serving where slot variables aren't needed.
</td>
</tr><tr>
<td>
`experimental_sparsecore_restore_info`<a id="experimental_sparsecore_restore_info"></a>
</td>
<td>
Information from the sparse core
training, required to restore from checkpoint for serving (like number
of TPU devices used `num_tpu_devices`.)
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Raises</h2></th></tr>

<tr>
<td>
`RuntimeError`<a id="RuntimeError"></a>
</td>
<td>
If created under TPUStrategy.
</td>
</tr>
</table>





<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`embedding_tables`<a id="embedding_tables"></a>
</td>
<td>
Returns a dict of embedding tables, keyed by `TableConfig`.
</td>
</tr>
</table>



## Methods

<h3 id="build"><code>build</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/tpu/tpu_embedding_base.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>build()
</code></pre>

Create variables and slots variables for TPU embeddings.


<h3 id="embedding_lookup"><code>embedding_lookup</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/tpu/tpu_embedding_for_serving.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>embedding_lookup(
    features: Any, weights: Optional[Any] = None
) -> Any
</code></pre>

Apply standard lookup ops on CPU.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`features`
</td>
<td>
A nested structure of <a href="../../../../tf/Tensor.md"><code>tf.Tensor</code></a>s, <a href="../../../../tf/sparse/SparseTensor.md"><code>tf.SparseTensor</code></a>s or
<a href="../../../../tf/RaggedTensor.md"><code>tf.RaggedTensor</code></a>s, with the same structure as `feature_config`. Inputs
will be downcast to <a href="../../../../tf.md#int32"><code>tf.int32</code></a>. Only one type out of <a href="../../../../tf/sparse/SparseTensor.md"><code>tf.SparseTensor</code></a>
or <a href="../../../../tf/RaggedTensor.md"><code>tf.RaggedTensor</code></a> is supported per call.
</td>
</tr><tr>
<td>
`weights`
</td>
<td>
If not `None`, a nested structure of <a href="../../../../tf/Tensor.md"><code>tf.Tensor</code></a>s,
<a href="../../../../tf/sparse/SparseTensor.md"><code>tf.SparseTensor</code></a>s or <a href="../../../../tf/RaggedTensor.md"><code>tf.RaggedTensor</code></a>s, matching the above, except
that the tensors should be of float type (and they will be downcast to
<a href="../../../../tf.md#float32"><code>tf.float32</code></a>). For <a href="../../../../tf/sparse/SparseTensor.md"><code>tf.SparseTensor</code></a>s we assume the `indices` are the
same for the parallel entries from `features` and similarly for
<a href="../../../../tf/RaggedTensor.md"><code>tf.RaggedTensor</code></a>s we assume the row_splits are the same.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A nested structure of Tensors with the same structure as input features.
</td>
</tr>

</table>



<h3 id="__call__"><code>__call__</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/tpu/tpu_embedding_base.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__call__(
    features: Any, weights: Optional[Any] = None
) -> Any
</code></pre>

Call the mid level api to do embedding lookup.




