description: Options related to the tf.data service cross trainer cache.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.data.experimental.service.CrossTrainerCache" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
</div>

# tf.data.experimental.service.CrossTrainerCache

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/data/experimental/ops/data_service_ops.py">View source</a>



Options related to the tf.data service cross trainer cache.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.data.experimental.service.CrossTrainerCache`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.data.experimental.service.CrossTrainerCache(
    trainer_id
)
</code></pre>



<!-- Placeholder for "Used in" -->

This is used to enable cross-trainer cache when distributing a dataset. For
example:

```
dataset = dataset.apply(tf.data.experimental.service.distribute(
    processing_mode=tf.data.experimental.service.ShardingPolicy.OFF,
    service=FLAGS.tf_data_service_address,
    job_name="job",
    cross_trainer_cache=data_service_ops.CrossTrainerCache(
        trainer_id=trainer_id())))
```

For more details, refer to
https://www.tensorflow.org/api_docs/python/tf/data/experimental/service#sharing_tfdata_service_with_concurrent_trainers.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`trainer_id`<a id="trainer_id"></a>
</td>
<td>
Each training job has a unique ID. Once a job has consumed
data, the data remains in the cache and is re-used by jobs with different
`trainer_id`s. Requests with the same `trainer_id` do not re-use data.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Raises</h2></th></tr>
<tr class="alt">
<td colspan="2">
ValueError if `trainer_id` is empty.
</td>
</tr>

</table>



