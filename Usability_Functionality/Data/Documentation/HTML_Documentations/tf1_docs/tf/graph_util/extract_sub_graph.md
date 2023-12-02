<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.graph_util.extract_sub_graph" />
<meta itemprop="path" content="Stable" />
</div>

# tf.graph_util.extract_sub_graph

Extract the subgraph that can reach any of the nodes in 'dest_nodes'. (deprecated)

### Aliases:

* `tf.compat.v1.graph_util.extract_sub_graph`
* `tf.compat.v2.compat.v1.graph_util.extract_sub_graph`
* `tf.graph_util.extract_sub_graph`

``` python
tf.graph_util.extract_sub_graph(
    graph_def,
    dest_nodes
)
```

<!-- Placeholder for "Used in" -->

Warning: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating:
Use <a href="../../tf/graph_util/extract_sub_graph.md"><code>tf.compat.v1.graph_util.extract_sub_graph</code></a>

#### Args:


* <b>`graph_def`</b>: A graph_pb2.GraphDef proto.
* <b>`dest_nodes`</b>: A list of strings specifying the destination node names.

#### Returns:

The GraphDef of the sub-graph.



#### Raises:


* <b>`TypeError`</b>: If 'graph_def' is not a graph_pb2.GraphDef proto.