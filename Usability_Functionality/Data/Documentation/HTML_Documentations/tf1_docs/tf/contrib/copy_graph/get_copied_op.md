<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.contrib.copy_graph.get_copied_op" />
<meta itemprop="path" content="Stable" />
</div>

# tf.contrib.copy_graph.get_copied_op

Given an `Operation` instance from some `Graph`, returns

``` python
tf.contrib.copy_graph.get_copied_op(
    org_instance,
    graph,
    scope=''
)
```

<!-- Placeholder for "Used in" -->
its namesake from `graph`, under the specified scope
(default `""`).

If a copy of `org_instance` is present in `graph` under the given
`scope`, it will be returned.

#### Args:


* <b>`org_instance`</b>: An `Operation` from some `Graph`.
* <b>`graph`</b>: The `Graph` to be searched for a copr of `org_instance`.
* <b>`scope`</b>: The scope `org_instance` is present in.


#### Returns:

The `Operation` copy from `graph`.
