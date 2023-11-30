description: Return the default mesh under the current dtensor device context.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.experimental.dtensor.get_default_mesh" />
<meta itemprop="path" content="Stable" />
</div>

# tf.experimental.dtensor.get_default_mesh

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/dtensor/python/api.py">View source</a>



Return the default mesh under the current dtensor device context.


<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.experimental.dtensor.get_default_mesh() -> Optional[<a href="../../../tf/experimental/dtensor/Mesh.md"><code>tf.experimental.dtensor.Mesh</code></a>]
</code></pre>



<!-- Placeholder for "Used in" -->

In the case that dtensor device system is not initialized, this function
will return None.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
The current default mesh for the dtensor device context.
</td>
</tr>

</table>

