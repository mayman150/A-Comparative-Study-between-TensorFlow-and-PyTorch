description: Gathers and serializes a trackable view.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.train.TrackableView" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="children"/>
<meta itemprop="property" content="descendants"/>
</div>

# tf.train.TrackableView

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/checkpoint/trackable_view.py">View source</a>



Gathers and serializes a trackable view.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.train.TrackableView(
    root
)
</code></pre>



<!-- Placeholder for "Used in" -->


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
>>> trackable_view = tf.train.TrackableView(root)
```

Pass root to tf.train.TrackableView.children() to get the dictionary of all
children directly linked to root by name.
```
>>> trackable_view_children = trackable_view.children(root)
>>> for item in trackable_view_children.items():
...   print(item)
('a_var', <tf.Variable 'Variable:0' shape=() dtype=float32, numpy=5.0>)
('b_var', <tf.Variable 'Variable:0' shape=() dtype=float32, numpy=4.0>)
('vars', ListWrapper([<tf.Variable 'Variable:0' shape=() dtype=float32,
numpy=1.0>, <tf.Variable 'Variable:0' shape=() dtype=float32, numpy=2.0>]))
('leaf', ...)
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`root`<a id="root"></a>
</td>
<td>
A `Trackable` object whose variables (including the variables of
dependencies, recursively) should be saved. May be a weak reference.
</td>
</tr>
</table>





<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`root`<a id="root"></a>
</td>
<td>

</td>
</tr>
</table>



## Methods

<h3 id="children"><code>children</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/checkpoint/trackable_view.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@classmethod</code>
<code>children(
    obj, save_type=base.SaveType.CHECKPOINT, **kwargs
)
</code></pre>

Returns all child trackables attached to obj.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`obj`
</td>
<td>
A `Trackable` object.
</td>
</tr><tr>
<td>
`save_type`
</td>
<td>
A string, can be 'savedmodel' or 'checkpoint'.
</td>
</tr><tr>
<td>
`**kwargs`
</td>
<td>
kwargs to use when retrieving the object's children.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
Dictionary of all children attached to the object with name to trackable.
</td>
</tr>

</table>



<h3 id="descendants"><code>descendants</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/checkpoint/trackable_view.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>descendants()
</code></pre>

Returns a list of all nodes from self.root using a breadth first traversal.




