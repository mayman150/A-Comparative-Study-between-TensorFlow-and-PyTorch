description: Fetches the layout of a DTensor.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.experimental.dtensor.fetch_layout" />
<meta itemprop="path" content="Stable" />
</div>

# tf.experimental.dtensor.fetch_layout

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/dtensor/python/api.py">View source</a>



Fetches the layout of a DTensor.


<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.experimental.dtensor.fetch_layout(
    tensor: <a href="../../../tf/Tensor.md"><code>tf.Tensor</code></a>
) -> <a href="../../../tf/experimental/dtensor/Layout.md"><code>tf.experimental.dtensor.Layout</code></a>
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
The DTensor whose layout is to be fetched.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
The `Layout` of this DTensor.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Raises</h2></th></tr>

<tr>
<td>
`RuntimeError`<a id="RuntimeError"></a>
</td>
<td>
When not called eagerly.
</td>
</tr>
</table>

