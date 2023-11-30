description: Asserts that the layout of the DTensor is layout.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.experimental.dtensor.check_layout" />
<meta itemprop="path" content="Stable" />
</div>

# tf.experimental.dtensor.check_layout

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/dtensor/python/api.py">View source</a>



Asserts that the layout of the DTensor is `layout`.


<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.experimental.dtensor.check_layout(
    tensor: <a href="../../../tf/Tensor.md"><code>tf.Tensor</code></a>,
    layout: <a href="../../../tf/experimental/dtensor/Layout.md"><code>tf.experimental.dtensor.Layout</code></a>
) -> None
</code></pre>



<!-- Placeholder for "Used in" -->


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`tensor`<a id="tensor"></a>
</td>
<td>
A DTensor whose layout is to be checked.
</td>
</tr><tr>
<td>
`layout`<a id="layout"></a>
</td>
<td>
The `Layout` to compare against.
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
If the layout of `tensor` does not match the supplied `layout`.
</td>
</tr>
</table>

