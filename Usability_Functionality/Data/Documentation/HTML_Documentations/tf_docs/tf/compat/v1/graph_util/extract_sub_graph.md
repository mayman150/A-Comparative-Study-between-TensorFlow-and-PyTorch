description: Extract the subgraph that can reach any of the nodes in 'dest_nodes'. (deprecated)

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.compat.v1.graph_util.extract_sub_graph" />
<meta itemprop="path" content="Stable" />
</div>

# tf.compat.v1.graph_util.extract_sub_graph

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/framework/graph_util_impl.py">View source</a>



Extract the subgraph that can reach any of the nodes in 'dest_nodes'. (deprecated)


<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.compat.v1.graph_util.extract_sub_graph(
    graph_def, dest_nodes
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
`graph_def`<a id="graph_def"></a>
</td>
<td>
A graph_pb2.GraphDef proto.
</td>
</tr><tr>
<td>
`dest_nodes`<a id="dest_nodes"></a>
</td>
<td>
An iterable of strings specifying the destination node names.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
The GraphDef of the sub-graph.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Raises</h2></th></tr>

<tr>
<td>
`TypeError`<a id="TypeError"></a>
</td>
<td>
If 'graph_def' is not a graph_pb2.GraphDef proto.
</td>
</tr>
</table>

