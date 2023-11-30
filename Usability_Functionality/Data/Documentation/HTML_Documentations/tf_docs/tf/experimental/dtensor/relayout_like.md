description: Changes the layout of tensor to the same as layout_tensor.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.experimental.dtensor.relayout_like" />
<meta itemprop="path" content="Stable" />
</div>

# tf.experimental.dtensor.relayout_like

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/dtensor/python/api.py">View source</a>



Changes the layout of `tensor` to the same as `layout_tensor`.


<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.experimental.dtensor.relayout_like(
    tensor: <a href="../../../tf/Tensor.md"><code>tf.Tensor</code></a>,
    layout_tensor: <a href="../../../tf/Tensor.md"><code>tf.Tensor</code></a>,
    name: Optional[str] = None
) -> <a href="../../../tf/Tensor.md"><code>tf.Tensor</code></a>
</code></pre>



<!-- Placeholder for "Used in" -->

`relayout_like` is often used inside a <a href="../../../tf/function.md"><code>tf.function</code></a>, to ensure a tensor is
placed to the same mesh and with the same layout as another tensor.

The backward gradient of a `relayout` is a `relayout_like` operation, to
ensure the backward tensor has the same layout as the forward input tensor:

```
@ops.RegisterGradient("Relayout")
def _relayout_gradient(op, grad):
  return relayout_like(grad, layout_input=op.inputs[0])
```

Here is another illustrative example:

```
@tf.function
def func(x):
  z = tf.ones(x.shape)
  z = dtensor.relayout_like(z, x)
  return x + z

with dtensor.default_mesh(cpu_mesh):
  x = tf.ones((4, 4))

with dtensor.default_mesh(gpu_mesh):
  y = func(x)

# y would be on the cpu mesh, following the mesh of x.
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`tensor`<a id="tensor"></a>
</td>
<td>
A DTensor to specify a new layout for.
</td>
</tr><tr>
<td>
`layout_tensor`<a id="layout_tensor"></a>
</td>
<td>
A Tensor object whose layout will be used for the layout of
result. The shape and type of layout_tensor are irrelevant.
</td>
</tr><tr>
<td>
`name`<a id="name"></a>
</td>
<td>
name of the Op.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A DTensor output from the RelayoutLike op.
</td>
</tr>

</table>

