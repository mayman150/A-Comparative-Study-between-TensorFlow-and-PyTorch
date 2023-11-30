description: Returns True if the given node_def must run on CPU, otherwise False. (deprecated)

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.compat.v1.graph_util.must_run_on_cpu" />
<meta itemprop="path" content="Stable" />
</div>

# tf.compat.v1.graph_util.must_run_on_cpu

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/framework/graph_util_impl.py">View source</a>



Returns True if the given node_def must run on CPU, otherwise False. (deprecated)


<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.compat.v1.graph_util.must_run_on_cpu(
    node, pin_variables_on_cpu=False
)
</code></pre>



<!-- Placeholder for "Used in" -->

Deprecated: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating:
This API was designed for TensorFlow v1. See https://www.tensorflow.org/guide/migrate for instructions on how to migrate your code to TensorFlow v2.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`node`<a id="node"></a>
</td>
<td>
The node to be assigned to a device. Could be either an ops.Operation
or NodeDef.
</td>
</tr><tr>
<td>
`pin_variables_on_cpu`<a id="pin_variables_on_cpu"></a>
</td>
<td>
If True, this function will return False if node_def
represents a variable-related op.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
True if the given node must run on CPU, otherwise False.
</td>
</tr>

</table>

