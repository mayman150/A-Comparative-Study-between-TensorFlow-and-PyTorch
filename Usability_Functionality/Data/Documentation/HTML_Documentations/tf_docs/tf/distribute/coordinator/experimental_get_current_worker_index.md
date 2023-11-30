description: Returns the current worker index, when called within a worker closure.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.distribute.coordinator.experimental_get_current_worker_index" />
<meta itemprop="path" content="Stable" />
</div>

# tf.distribute.coordinator.experimental_get_current_worker_index

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/distribute/coordinator/coordinator_context.py">View source</a>



Returns the current worker index, when called within a worker closure.


<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.distribute.coordinator.experimental_get_current_worker_index()
</code></pre>



<!-- Placeholder for "Used in" -->

Some parameter server training workloads may require the worker to know its
index, for example for data sharding for reduced-variance training.

This method may be used within a <a href="../../../tf/function.md"><code>tf.function</code></a> that is executed on a worker.
That is, either a `dataset_fn` that runs via
<a href="../../../tf/distribute/experimental/coordinator/ClusterCoordinator.md#create_per_worker_dataset"><code>ClusterCoordinator.create_per_worker_dataset</code></a>, or any other function
scheduled via <a href="../../../tf/distribute/experimental/coordinator/ClusterCoordinator.md#schedule"><code>ClusterCoordinator.schedule</code></a>.

Example (sharding data by worker):

```python
strategy = tf.distribute.ParameterServerStrategy(
    cluster_resolver=...)
coordinator = (
    tf.distribute.coordinator.ClusterCoordinator(strategy))

def dataset_fn(context):
  dataset = tf.data.Dataset.range(10)
  worker_index = (
      tf.distribute.coordinator.experimental_get_current_worker_index()
  )
  dataset = dataset.shard(
      num_shards=num_workers,
      index=worker_index,
  )
  return dataset

@tf.function
def per_worker_dataset_fn():
  return strategy.distribute_datasets_from_function(dataset_fn)

per_worker_dataset = coordinator.create_per_worker_dataset(
    per_worker_dataset_fn)
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Raises</h2></th></tr>

<tr>
<td>
`RuntimeError`<a id="RuntimeError"></a>
</td>
<td>
if called from outside a <a href="../../../tf/function.md"><code>tf.function</code></a> or outside of a remote
closure execution context (that is, on a non-worker machine).
</td>
</tr>
</table>

