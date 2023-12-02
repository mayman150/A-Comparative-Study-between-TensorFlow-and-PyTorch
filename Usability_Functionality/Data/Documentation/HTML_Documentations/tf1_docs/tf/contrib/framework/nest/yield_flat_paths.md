<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.contrib.framework.nest.yield_flat_paths" />
<meta itemprop="path" content="Stable" />
</div>

# tf.contrib.framework.nest.yield_flat_paths

Yields paths for some nested structure.

``` python
tf.contrib.framework.nest.yield_flat_paths(
    nest,
    expand_composites=False
)
```

<!-- Placeholder for "Used in" -->

Paths are lists of objects which can be str-converted, which may include
integers or other types which are used as indices in a dict.

The flat list will be in the corresponding order as if you called
`snt.nest.flatten` on the structure. This is handy for naming Tensors such
the TF scope structure matches the tuple structure.

E.g. if we have a tuple `value = Foo(a=3, b=Bar(c=23, d=42))`

```shell
>>> nest.flatten(value)
[3, 23, 42]
>>> list(nest.yield_flat_paths(value))
[('a',), ('b', 'c'), ('b', 'd')]
```

```shell
>>> list(nest.yield_flat_paths({'a': [3]}))
[('a', 0)]
>>> list(nest.yield_flat_paths({'a': 3}))
[('a',)]
```

#### Args:


* <b>`nest`</b>: the value to produce a flattened paths list for.
* <b>`expand_composites`</b>: If true, then composite tensors such as tf.SparseTensor
   and tf.RaggedTensor are expanded into their component tensors.


#### Yields:

Tuples containing index or key values which form the path to a specific
  leaf value in the nested structure.
