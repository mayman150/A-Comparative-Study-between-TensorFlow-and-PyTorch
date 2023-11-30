description: Creates a dataset that reads data from the tf.data service.
robots: noindex

# tf.raw_ops.DataServiceDatasetV3

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>



Creates a dataset that reads data from the tf.data service.


<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.raw_ops.DataServiceDatasetV3`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.raw_ops.DataServiceDatasetV3(
    dataset_id,
    processing_mode,
    address,
    protocol,
    job_name,
    consumer_index,
    num_consumers,
    max_outstanding_requests,
    iteration_counter,
    output_types,
    output_shapes,
    uncompress_fn,
    task_refresh_interval_hint_ms=-1,
    data_transfer_protocol=&#x27;&#x27;,
    target_workers=&#x27;AUTO&#x27;,
    uncompress=False,
    cross_trainer_cache_options=&#x27;&#x27;,
    name=None
)
</code></pre>



<!-- Placeholder for "Used in" -->


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`dataset_id`<a id="dataset_id"></a>
</td>
<td>
A `Tensor` of type `int64`.
</td>
</tr><tr>
<td>
`processing_mode`<a id="processing_mode"></a>
</td>
<td>
A `Tensor` of type `string`.
</td>
</tr><tr>
<td>
`address`<a id="address"></a>
</td>
<td>
A `Tensor` of type `string`.
</td>
</tr><tr>
<td>
`protocol`<a id="protocol"></a>
</td>
<td>
A `Tensor` of type `string`.
</td>
</tr><tr>
<td>
`job_name`<a id="job_name"></a>
</td>
<td>
A `Tensor` of type `string`.
</td>
</tr><tr>
<td>
`consumer_index`<a id="consumer_index"></a>
</td>
<td>
A `Tensor` of type `int64`.
</td>
</tr><tr>
<td>
`num_consumers`<a id="num_consumers"></a>
</td>
<td>
A `Tensor` of type `int64`.
</td>
</tr><tr>
<td>
`max_outstanding_requests`<a id="max_outstanding_requests"></a>
</td>
<td>
A `Tensor` of type `int64`.
</td>
</tr><tr>
<td>
`iteration_counter`<a id="iteration_counter"></a>
</td>
<td>
A `Tensor` of type `resource`.
</td>
</tr><tr>
<td>
`output_types`<a id="output_types"></a>
</td>
<td>
A list of `tf.DTypes` that has length `>= 1`.
</td>
</tr><tr>
<td>
`output_shapes`<a id="output_shapes"></a>
</td>
<td>
A list of shapes (each a <a href="../../tf/TensorShape.md"><code>tf.TensorShape</code></a> or list of `ints`) that has length `>= 1`.
</td>
</tr><tr>
<td>
`uncompress_fn`<a id="uncompress_fn"></a>
</td>
<td>
A function decorated with @Defun.
</td>
</tr><tr>
<td>
`task_refresh_interval_hint_ms`<a id="task_refresh_interval_hint_ms"></a>
</td>
<td>
An optional `int`. Defaults to `-1`.
</td>
</tr><tr>
<td>
`data_transfer_protocol`<a id="data_transfer_protocol"></a>
</td>
<td>
An optional `string`. Defaults to `""`.
</td>
</tr><tr>
<td>
`target_workers`<a id="target_workers"></a>
</td>
<td>
An optional `string`. Defaults to `"AUTO"`.
</td>
</tr><tr>
<td>
`uncompress`<a id="uncompress"></a>
</td>
<td>
An optional `bool`. Defaults to `False`.
</td>
</tr><tr>
<td>
`cross_trainer_cache_options`<a id="cross_trainer_cache_options"></a>
</td>
<td>
An optional `string`. Defaults to `""`.
</td>
</tr><tr>
<td>
`name`<a id="name"></a>
</td>
<td>
A name for the operation (optional).
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A `Tensor` of type `variant`.
</td>
</tr>

</table>

