description: The TPUEmbedding mid level API running on TPU without Embedding accelerator.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.tpu.experimental.embedding.TPUEmbeddingV0" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__call__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="build"/>
<meta itemprop="property" content="embedding_lookup"/>
</div>

# tf.tpu.experimental.embedding.TPUEmbeddingV0

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/tpu/tpu_embedding_v1.py">View source</a>



The TPUEmbedding mid level API running on TPU without Embedding accelerator.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.tpu.experimental.embedding.TPUEmbeddingV0`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.tpu.experimental.embedding.TPUEmbeddingV0(
    feature_config: Union[<a href="../../../../tf/tpu/experimental/embedding/FeatureConfig.md"><code>tf.tpu.experimental.embedding.FeatureConfig</code></a>, Iterable],
    optimizer: Optional[tpu_embedding_v2_utils._Optimizer]
)
</code></pre>



<!-- Placeholder for "Used in" -->

NOTE: This mid level API is not intended for large embedding table lookup.
Embedding tables will be replicated across devices rather than sharding
across them. To do large embedding table lookup, please use the
<a href="../../../../tf/tpu/experimental/embedding/TPUEmbedding.md"><code>tpu.experimental.embedding.TPUEmbedding</code></a> class. This class is an alternative
way to do embedding lookups when the TPU doesn't support any version of
embedding feature. See
`tpu.experimental.tpu_hardware_feature.embedding_feature` for a detailed
explanation.

This class has to be created under the `TPUStrategy`, Otherwise a RuntimeError
will be raised.
```python
strategy = tf.distribute.TPUStrategy(...)
with strategy.scope():
  embedding = tf.tpu.experimental.embedding.TPUEmbeddingV0(
      feature_config=feature_config,
      optimizer=tf.tpu.experimental.embedding.SGD(0.1))
```
When creating a distributed dataset that is to be passed to the lookup
operation a special input option must be specified:

```python
distributed_dataset = (
    strategy.distribute_datasets_from_function(
        dataset_fn=...,
        options=tf.distribute.InputOptions(
            experimental_fetch_to_device=False))
dataset_iterator = iter(distributed_dataset)
```

Below is an example of a training and evaluation step:

```python
optimizer = tf.keras.optimizers.SGD(0.1)

@tf.function
def training_step(dataset_iterator, num_steps):
  def tpu_step(embedding_features):
    with tf.GradientTape() as tape:
      tape.watch(embedding.embedding_table.values())
      activation = embedding(embedding_features)
      model_output = model(activations)
      loss = ...  # some function of labels and model_output

    embedding_gradients = tape.gradient(loss,
                                        embedding.embedding_table.values())
    optimizer.apply_gradients(list(zip(gradients,
                              mid_level_api.embedding_tables.values())))
    # Insert your model gradient and optimizer application here

  for _ in tf.range(num_steps):
    strategy.run(tpu_step, args=(next(dataset_iterator), ))

@tf.function
def evalution_step(dataset_iterator, num_steps):
  def tpu_step(embedding_features):
    activations = embedding(embedding_features)
    model_output = model(activations)
    # Insert your evaluation code here.

  for _ in tf.range(num_steps):
    strategy.run(tpu_step, args=(next(dataset_iterator), ))
```

NOTE: The optimizer used here is a Keras optimizer. In order to make the slot
variable creation stay consistent between Keras optimizers and
embedding optimizers, the `slot_variable_creation_fn` argument of the
embedding optimizers has to be passed with the Keras `add_slot` function. Also
note that the slot names might be slightly different between them.

```python
optimizer = tf.keras.optimizers.Adagrad(learning_rate=0.1)

def slot_variable_creation_fn(table, slot_names, slot_initializers):
    slots = {}
    for slot, initializer in zip(slot_names, slot_initializers):
      slots[slot] = optimizer.add_slot(table, slot, initializer)
    return slots

embedding_optimizer = tf.experimental.embedding.Adagrad(
    learning_rate=0.1,
    slot_variable_creation_fn=slot_variable_creation_fn)

# Use the embedding optimizer to create mid level api and keras optimizer to
# apply gradients.
```



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

<a target="_blank" class="external" href="/code/stable/tensorflow/python/tpu/tpu_embedding_v1.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>embedding_lookup(
    features: Any, weights: Optional[Any] = None
) -> Any
</code></pre>

Apply embedding lookup on TPUs using Tensorcore.

Note that all the sparse and ragged tensors will be converted to dense
tensors on CPU and then passed to the TPU to do embedding look up. Large
embedding lookup is not supported by this API, use the TPUEmbedding mid
level api instead.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`features`
</td>
<td>
a nested structure of Tensors, SparseTensors or RaggedTensors.
</td>
</tr><tr>
<td>
`weights`
</td>
<td>
a nested structure of Tensors, SparseTensors or RaggedTensors or
None for no weights. If not None, structure must match that of inputs,
but entries are allowed to be None.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A nested structure of Tensors with the same structure as inputs.
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




