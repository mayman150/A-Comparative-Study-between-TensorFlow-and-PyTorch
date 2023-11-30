description: Check whether the input tensor is a DTensor.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.experimental.dtensor.is_dtensor" />
<meta itemprop="path" content="Stable" />
</div>

# tf.experimental.dtensor.is_dtensor

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/dtensor/python/api.py">View source</a>



Check whether the input tensor is a DTensor.


<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.experimental.dtensor.is_dtensor(
    tensor
) -> bool
</code></pre>



<!-- Placeholder for "Used in" -->

In Python, a DTensor has the same type as a <a href="../../../tf/Tensor.md"><code>tf.Tensor</code></a>. This method will
let you check and handle the tensor differently if a tf.Tensor is a DTensor.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`tensor`<a id="tensor"></a>
</td>
<td>
an object to be checked.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
bool, True if the given tensor is a DTensor.
</td>
</tr>

</table>

