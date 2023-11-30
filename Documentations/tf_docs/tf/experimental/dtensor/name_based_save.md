description: Saves name based Tensor into a Checkpoint.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.experimental.dtensor.name_based_save" />
<meta itemprop="path" content="Stable" />
</div>

# tf.experimental.dtensor.name_based_save

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/dtensor/python/save_restore.py">View source</a>



Saves name based Tensor into a Checkpoint.


<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.experimental.dtensor.name_based_save(
    mesh: <a href="../../../tf/experimental/dtensor/Mesh.md"><code>tf.experimental.dtensor.Mesh</code></a>,
    checkpoint_prefix: Union[str, <a href="../../../tf/Tensor.md"><code>tf.Tensor</code></a>],
    name_tensor_dict: Dict[str, Union[tensor_lib.Tensor, tf_variables.Variable]]
)
</code></pre>



<!-- Placeholder for "Used in" -->

The function prepares the input dictionary to the format of a `sharded_save`,
so that it can take advantage of DTensor SPMD based distributed save.

Same as restore, the function only supports saving on the single mesh.

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

