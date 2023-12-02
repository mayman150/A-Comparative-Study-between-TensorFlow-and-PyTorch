<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.contrib.graph_editor.get_tensors" />
<meta itemprop="path" content="Stable" />
</div>

# tf.contrib.graph_editor.get_tensors

get all the tensors which are input or output of an op in the graph.

``` python
tf.contrib.graph_editor.get_tensors(graph)
```

<!-- Placeholder for "Used in" -->


#### Args:


* <b>`graph`</b>: a <a href="../../../tf/Graph.md"><code>tf.Graph</code></a>.

#### Returns:

A list of <a href="../../../tf/Tensor.md"><code>tf.Tensor</code></a>.


#### Raises:


* <b>`TypeError`</b>: if graph is not a <a href="../../../tf/Graph.md"><code>tf.Graph</code></a>.