description: Saves given named tensor slices in a sharded, multi-client safe fashion.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.experimental.dtensor.sharded_save" />
<meta itemprop="path" content="Stable" />
</div>

# tf.experimental.dtensor.sharded_save

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/dtensor/python/save_restore.py">View source</a>



Saves given named tensor slices in a sharded, multi-client safe fashion.


<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.experimental.dtensor.sharded_save(
    mesh: <a href="../../../tf/experimental/dtensor/Mesh.md"><code>tf.experimental.dtensor.Mesh</code></a>,
    file_prefix: Union[str, <a href="../../../tf/Tensor.md"><code>tf.Tensor</code></a>],
    tensor_names: Union[List[str], <a href="../../../tf/Tensor.md"><code>tf.Tensor</code></a>],
    shape_and_slices: Union[List[str], <a href="../../../tf/Tensor.md"><code>tf.Tensor</code></a>],
    tensors: List[Union[tensor_lib.Tensor, tf_variables.Variable]]
)
</code></pre>



<!-- Placeholder for "Used in" -->

The method makes sure the checkpoint directory state is correct in a sharded
mutli-client saving. Namely, we place a barrier after SaveV2 to make sure
every client has done writing the files. And another one after
MergeV2Checkpoints to make sure all Metadata is properly merged.

Upon existing, the checkpoint is completed and the all directory operations
are done.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`mesh`<a id="mesh"></a>
</td>
<td>
The Mesh that contains the Tensors to save.
</td>
</tr><tr>
<td>
`file_prefix`<a id="file_prefix"></a>
</td>
<td>
The prefix of checkpoint.
</td>
</tr><tr>
<td>
`tensor_names`<a id="tensor_names"></a>
</td>
<td>
a list of tensor names used in save op.
</td>
</tr><tr>
<td>
`shape_and_slices`<a id="shape_and_slices"></a>
</td>
<td>
a list of shape and slice specification used in save op.
The only supported value is "" as we don't support distributed saving with
slices yet.
</td>
</tr><tr>
<td>
`tensors`<a id="tensors"></a>
</td>
<td>
a list of tensors used in save op. The order should match
tensor_names.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A MergeV2Checkpoints op that merged all Metadata.
</td>
</tr>

</table>

