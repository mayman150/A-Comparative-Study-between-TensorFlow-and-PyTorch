description: Sets the default DTensor device mesh to use for enclosed functions.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.experimental.dtensor.default_mesh" />
<meta itemprop="path" content="Stable" />
</div>

# tf.experimental.dtensor.default_mesh

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/dtensor/python/api.py">View source</a>



Sets the default DTensor device mesh to use for enclosed functions.


<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@contextlib.contextmanager</code>
<code>tf.experimental.dtensor.default_mesh(
    mesh: <a href="../../../tf/experimental/dtensor/Mesh.md"><code>tf.experimental.dtensor.Mesh</code></a>
)
</code></pre>



<!-- Placeholder for "Used in" -->

This function returns a scope. All the ops and tf.functions in this scope will
default to this DTensor mesh if a mesh cannot be inferred from any of the
inputs
This is useful for wrapping any tf.function that doesn't take a DTensor as
input but would like to produce DTensor as result. The scope will also make
sure all small constants are replicated as DTensors.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`mesh`<a id="mesh"></a>
</td>
<td>
A Mesh instance to extract a default mesh from.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Yields</h2></th></tr>
<tr class="alt">
<td colspan="2">
A context in which all ops and tf.functions will run on the given mesh.
</td>
</tr>

</table>

