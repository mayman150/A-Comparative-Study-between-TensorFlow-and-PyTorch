description: Restores from checkpoint_prefix to name based DTensors.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.experimental.dtensor.name_based_restore" />
<meta itemprop="path" content="Stable" />
</div>

# tf.experimental.dtensor.name_based_restore

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/dtensor/python/save_restore.py">View source</a>



Restores from checkpoint_prefix to name based DTensors.


<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.experimental.dtensor.name_based_restore(
    mesh: <a href="../../../tf/experimental/dtensor/Mesh.md"><code>tf.experimental.dtensor.Mesh</code></a>,
    checkpoint_prefix: str,
    name_tensor_dict: Dict[str, Union[tensor_lib.Tensor, tf_variables.Variable]]
)
</code></pre>



<!-- Placeholder for "Used in" -->

It is required to have already-initialized DTensor variables that have same
shape/dtype for the tensors being restored.

Also, we currently only support a named based restore on a single mesh.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`mesh`<a id="mesh"></a>
</td>
<td>
The single mesh that all Tensors would be restored to.
</td>
</tr><tr>
<td>
`checkpoint_prefix`<a id="checkpoint_prefix"></a>
</td>
<td>
The prefix of checkpoint to be restored.
</td>
</tr><tr>
<td>
`name_tensor_dict`<a id="name_tensor_dict"></a>
</td>
<td>
A ordered dictionary of tensor_names to a DTensor. The
DTensor shape/dtype must match the tensors being saved/restored for now.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A dictionary of name to its restored DTensor value.
</td>
</tr>

</table>

