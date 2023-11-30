description: Create a list of partitioned variables according to the given slicing. (deprecated)

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.compat.v1.create_partitioned_variables" />
<meta itemprop="path" content="Stable" />
</div>

# tf.compat.v1.create_partitioned_variables

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/ops/partitioned_variables.py">View source</a>



Create a list of partitioned variables according to the given `slicing`. (deprecated)


<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.compat.v1.create_partitioned_variables(
    shape,
    slicing,
    initializer,
    dtype=<a href="../../../tf/dtypes.md#float32"><code>tf.dtypes.float32</code></a>,
    trainable=True,
    collections=None,
    name=None,
    reuse=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

Deprecated: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating:
Use `tf.get_variable` with a partitioner set.

Currently only one dimension of the full variable can be sliced, and the
full variable can be reconstructed by the concatenation of the returned
list along that dimension.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`shape`<a id="shape"></a>
</td>
<td>
List of integers.  The shape of the full variable.
</td>
</tr><tr>
<td>
`slicing`<a id="slicing"></a>
</td>
<td>
List of integers.  How to partition the variable.
Must be of the same length as `shape`.  Each value
indicate how many slices to create in the corresponding
dimension.  Presently only one of the values can be more than 1;
that is, the variable can only be sliced along one dimension.

For convenience, The requested number of partitions does not have to
divide the corresponding dimension evenly.  If it does not, the
shapes of the partitions are incremented by 1 starting from partition
0 until all slack is absorbed.  The adjustment rules may change in the
future, but as you can save/restore these variables with different
slicing specifications this should not be a problem.
</td>
</tr><tr>
<td>
`initializer`<a id="initializer"></a>
</td>
<td>
A `Tensor` of shape `shape` or a variable initializer
function.  If a function, it will be called once for each slice,
passing the shape and data type of the slice as parameters.  The
function must return a tensor with the same shape as the slice.
</td>
</tr><tr>
<td>
`dtype`<a id="dtype"></a>
</td>
<td>
Type of the variables. Ignored if `initializer` is a `Tensor`.
</td>
</tr><tr>
<td>
`trainable`<a id="trainable"></a>
</td>
<td>
If True also add all the variables to the graph collection
`GraphKeys.TRAINABLE_VARIABLES`.
</td>
</tr><tr>
<td>
`collections`<a id="collections"></a>
</td>
<td>
List of graph collections keys to add the variables to.
Defaults to `[GraphKeys.GLOBAL_VARIABLES]`.
</td>
</tr><tr>
<td>
`name`<a id="name"></a>
</td>
<td>
Optional name for the full variable.  Defaults to
`"PartitionedVariable"` and gets uniquified automatically.
</td>
</tr><tr>
<td>
`reuse`<a id="reuse"></a>
</td>
<td>
Boolean or `None`; if `True` and name is set, it would reuse
previously created variables. if `False` it will create new variables.
if `None`, it would inherit the parent scope reuse.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A list of Variables corresponding to the slicing.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Raises</h2></th></tr>

<tr>
<td>
`ValueError`<a id="ValueError"></a>
</td>
<td>
If any of the arguments is malformed.
</td>
</tr>
</table>

