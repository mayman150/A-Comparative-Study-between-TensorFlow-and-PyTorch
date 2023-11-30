description: Gathers and serializes a checkpoint view.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.train.CheckpointView" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="children"/>
<meta itemprop="property" content="descendants"/>
<meta itemprop="property" content="diff"/>
<meta itemprop="property" content="match"/>
</div>

# tf.train.CheckpointView

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/checkpoint/checkpoint_view.py">View source</a>



Gathers and serializes a checkpoint view.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.train.CheckpointView(
    save_path
)
</code></pre>



<!-- Placeholder for "Used in" -->

This is for loading specific portions of a module from a
checkpoint, and be able to compare two modules by matching components.

#### Example usage:



```
>>> class SimpleModule(tf.Module):
...   def __init__(self, name=None):
...     super().__init__(name=name)
...     self.a_var = tf.Variable(5.0)
...     self.b_var = tf.Variable(4.0)
...     self.vars = [tf.Variable(1.0), tf.Variable(2.0)]
```

```
>>> root = SimpleModule(name="root")
>>> root.leaf = SimpleModule(name="leaf")
>>> ckpt = tf.train.Checkpoint(root)
>>> save_path = ckpt.save('/tmp/tf_ckpts')
>>> checkpoint_view = tf.train.CheckpointView(save_path)
```

Pass `node_id=0` to <a href="../../tf/train/CheckpointView.md#children"><code>tf.train.CheckpointView.children()</code></a> to get the dictionary
of all children directly linked to the checkpoint root.

```
>>> for name, node_id in checkpoint_view.children(0).items():
...   print(f"- name: '{name}', node_id: {node_id}")
- name: 'a_var', node_id: 1
- name: 'b_var', node_id: 2
- name: 'vars', node_id: 3
- name: 'leaf', node_id: 4
- name: 'root', node_id: 0
- name: 'save_counter', node_id: 5
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`save_path`<a id="save_path"></a>
</td>
<td>
The path to the checkpoint.
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
If the save_path does not lead to a TF2 checkpoint.
</td>
</tr>
</table>



## Methods

<h3 id="children"><code>children</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/checkpoint/checkpoint_view.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>children(
    node_id
)
</code></pre>

Returns all child trackables attached to obj.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`node_id`
</td>
<td>
Id of the node to return its children.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
Dictionary of all children attached to the object with name to node_id.
</td>
</tr>

</table>



<h3 id="descendants"><code>descendants</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/checkpoint/checkpoint_view.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>descendants()
</code></pre>

Returns a list of trackables by node_id attached to obj.


<h3 id="diff"><code>diff</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/checkpoint/checkpoint_view.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>diff(
    obj
)
</code></pre>

Returns diff between CheckpointView and Trackable.

This method is intended to be used to compare the object stored in a
checkpoint vs a live model in Python. For example, if checkpoint
restoration fails the `assert_consumed()` or
`assert_existing_objects_matched()` checks, you can use this to list out
the objects/checkpoint nodes which were not restored.

#### Example Usage:



```
>>> class SimpleModule(tf.Module):
...   def __init__(self, name=None):
...     super().__init__(name=name)
...     self.a_var = tf.Variable(5.0)
...     self.b_var = tf.Variable(4.0)
...     self.vars = [tf.Variable(1.0), tf.Variable(2.0)]
```

```
>>> root = SimpleModule(name="root")
>>> leaf = root.leaf = SimpleModule(name="leaf")
>>> leaf.leaf3 = tf.Variable(6.0, name="leaf3")
>>> leaf.leaf4 = tf.Variable(7.0, name="leaf4")
>>> ckpt = tf.train.Checkpoint(root)
>>> save_path = ckpt.save('/tmp/tf_ckpts')
>>> checkpoint_view = tf.train.CheckpointView(save_path)
```

```
>>> root2 = SimpleModule(name="root")
>>> leaf2 = root2.leaf2 = SimpleModule(name="leaf2")
>>> leaf2.leaf3 = tf.Variable(6.0)
>>> leaf2.leaf4 = tf.Variable(7.0)
```

Pass `node_id=0` to <a href="../../tf/train/CheckpointView.md#children"><code>tf.train.CheckpointView.children()</code></a> to get the
dictionary of all children directly linked to the checkpoint root.

```
>>> checkpoint_view_diff = checkpoint_view.diff(root2)
>>> checkpoint_view_match = checkpoint_view_diff[0].items()
>>> for item in checkpoint_view_match:
...   print(item)
(0, ...)
(1, <tf.Variable 'Variable:0' shape=() dtype=float32, numpy=5.0>)
(2, <tf.Variable 'Variable:0' shape=() dtype=float32, numpy=4.0>)
(3, ListWrapper([<tf.Variable 'Variable:0' shape=() dtype=float32,
numpy=1.0>, <tf.Variable 'Variable:0' shape=() dtype=float32, numpy=2.0>]))
(6, <tf.Variable 'Variable:0' shape=() dtype=float32, numpy=1.0>)
(7, <tf.Variable 'Variable:0' shape=() dtype=float32, numpy=2.0>)
```

```
>>> only_in_checkpoint_view = checkpoint_view_diff[1]
>>> print(only_in_checkpoint_view)
[4, 5, 8, 9, 10, 11, 12, 13, 14]
```

```
>>> only_in_trackable = checkpoint_view_diff[2]
>>> print(only_in_trackable)
[..., <tf.Variable 'Variable:0' shape=() dtype=float32, numpy=5.0>,
<tf.Variable 'Variable:0' shape=() dtype=float32, numpy=4.0>,
ListWrapper([<tf.Variable 'Variable:0' shape=() dtype=float32, numpy=1.0>,
<tf.Variable 'Variable:0' shape=() dtype=float32, numpy=2.0>]),
<tf.Variable 'Variable:0' shape=() dtype=float32, numpy=6.0>,
<tf.Variable 'Variable:0' shape=() dtype=float32, numpy=7.0>,
<tf.Variable 'Variable:0' shape=() dtype=float32, numpy=1.0>,
<tf.Variable 'Variable:0' shape=() dtype=float32, numpy=2.0>]
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`obj`
</td>
<td>
`Trackable` root.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
Tuple of (
- Overlaps: Dictionary containing all overlapping trackables that maps
`node_id` to `Trackable`, same as CheckpointView.match().
- Only in CheckpointView: List of `node_id` that only exist in
CheckpointView.
- Only in Trackable: List of `Trackable` that only exist in Trackable.
)
</td>
</tr>

</table>



<h3 id="match"><code>match</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/checkpoint/checkpoint_view.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>match(
    obj
)
</code></pre>

Returns all matching trackables between CheckpointView and Trackable.

Matching trackables represents trackables with the same name and position in
graph.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`obj`
</td>
<td>
`Trackable` root.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
Dictionary containing all overlapping trackables that maps `node_id` to
`Trackable`.
</td>
</tr>

</table>



#### Example usage:



```
>>> class SimpleModule(tf.Module):
...   def __init__(self, name=None):
...     super().__init__(name=name)
...     self.a_var = tf.Variable(5.0)
...     self.b_var = tf.Variable(4.0)
...     self.vars = [tf.Variable(1.0), tf.Variable(2.0)]
```

```
>>> root = SimpleModule(name="root")
>>> leaf = root.leaf = SimpleModule(name="leaf")
>>> leaf.leaf3 = tf.Variable(6.0, name="leaf3")
>>> leaf.leaf4 = tf.Variable(7.0, name="leaf4")
>>> ckpt = tf.train.Checkpoint(root)
>>> save_path = ckpt.save('/tmp/tf_ckpts')
>>> checkpoint_view = tf.train.CheckpointView(save_path)
```

```
>>> root2 = SimpleModule(name="root")
>>> leaf2 = root2.leaf2 = SimpleModule(name="leaf2")
>>> leaf2.leaf3 = tf.Variable(6.0)
>>> leaf2.leaf4 = tf.Variable(7.0)
```

Pass `node_id=0` to <a href="../../tf/train/CheckpointView.md#children"><code>tf.train.CheckpointView.children()</code></a> to get the
dictionary of all children directly linked to the checkpoint root.

```
>>> checkpoint_view_match = checkpoint_view.match(root2).items()
>>> for item in checkpoint_view_match:
...   print(item)
(0, ...)
(1, <tf.Variable 'Variable:0' shape=() dtype=float32, numpy=5.0>)
(2, <tf.Variable 'Variable:0' shape=() dtype=float32, numpy=4.0>)
(3, ListWrapper([<tf.Variable 'Variable:0' shape=() dtype=float32,
numpy=1.0>, <tf.Variable 'Variable:0' shape=() dtype=float32, numpy=2.0>]))
(6, <tf.Variable 'Variable:0' shape=() dtype=float32, numpy=1.0>)
(7, <tf.Variable 'Variable:0' shape=() dtype=float32, numpy=2.0>)
```



