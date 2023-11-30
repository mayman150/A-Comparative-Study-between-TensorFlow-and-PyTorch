description: Changes the layout of tensor.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.experimental.dtensor.relayout" />
<meta itemprop="path" content="Stable" />
</div>

# tf.experimental.dtensor.relayout

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/dtensor/python/api.py">View source</a>



Changes the layout of `tensor`.


<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.experimental.dtensor.relayout(
    tensor: <a href="../../../tf/Tensor.md"><code>tf.Tensor</code></a>,
    layout: <a href="../../../tf/experimental/dtensor/Layout.md"><code>tf.experimental.dtensor.Layout</code></a>,
    name: Optional[str] = None
) -> <a href="../../../tf/Tensor.md"><code>tf.Tensor</code></a>
</code></pre>



<!-- Placeholder for "Used in" -->

Changes the layout of `tensor` to `layout`. This is used to fine-tune the
behavior of ops following/connected to `tensor`, such as choosing one SPMD
expansion pattern over another. This works by forward propagating `layout`
to connected TensorFlow computation graphs during layout propagation.

Currently, only converting layouts from replicated to sharded or sharded to
replicated per mesh dimension is supported. That is, "x, y" -> "unsharded, y"
is supported, while "x, y" -> "z, y" is not supported.

We also support a special "match" sharding spec, which instructs the relayout
to act as an identity operation with respect to any sharding on these
mesh dimensions.

Relayout is internally lowered to a set of Split and/or AllToAll ops. When
tensor layouts are converted from replicated to sharded, the cost is
comparatively low because we only insert Split ops and no cross-device
communication is needed. However, when tensor layouts are converted from
sharded to replicated, cross-device communication may occur, causing potential
performance impact.

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
`layout`<a id="layout"></a>
</td>
<td>
A Layout object specifying a new sharding spec.
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
A DTensor output from the Relayout op.
</td>
</tr>

</table>

