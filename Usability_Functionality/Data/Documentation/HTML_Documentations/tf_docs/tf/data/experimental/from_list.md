description: Creates a Dataset comprising the given list of elements.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.data.experimental.from_list" />
<meta itemprop="path" content="Stable" />
</div>

# tf.data.experimental.from_list

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/data/experimental/ops/from_list.py">View source</a>



Creates a `Dataset` comprising the given list of elements.


<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.data.experimental.from_list`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.data.experimental.from_list(
    elements, name=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

The returned dataset will produce the items in the list one by one. The
functionality is identical to <a href="../../../tf/data/Dataset.md#from_tensor_slices"><code>Dataset.from_tensor_slices</code></a> when elements are
scalars, but different when elements have structure. Consider the following
example.

```
>>> dataset = tf.data.experimental.from_list([(1, 'a'), (2, 'b'), (3, 'c')])
>>> list(dataset.as_numpy_iterator())
[(1, b'a'), (2, b'b'), (3, b'c')]
```

To get the same output with `from_tensor_slices`, the data needs to be
reorganized:

```
>>> dataset = tf.data.Dataset.from_tensor_slices(([1, 2, 3], ['a', 'b', 'c']))
>>> list(dataset.as_numpy_iterator())
[(1, b'a'), (2, b'b'), (3, b'c')]
```

Unlike `from_tensor_slices`, `from_list` supports non-rectangular input:

```
>>> dataset = tf.data.experimental.from_list([[1], [2, 3]])
>>> list(dataset.as_numpy_iterator())
[array([1], dtype=int32), array([2, 3], dtype=int32)]
```

Achieving the same with `from_tensor_slices` requires the use of ragged
tensors.

`from_list` can be more performant than `from_tensor_slices` in some cases,
since it avoids the need for data slicing each epoch. However, it can also be
less performant, because data is stored as many small tensors rather than a
few large tensors as in `from_tensor_slices`. The general guidance is to
prefer `from_list` from a performance perspective when the number of elements
is small (less than 1000).

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`elements`<a id="elements"></a>
</td>
<td>
A list of elements whose components have the same nested
structure.
</td>
</tr><tr>
<td>
`name`<a id="name"></a>
</td>
<td>
(Optional.) A name for the tf.data operation.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>

<tr>
<td>
`Dataset`<a id="Dataset"></a>
</td>
<td>
A `Dataset` of the `elements`.
</td>
</tr>
</table>

