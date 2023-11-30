description: Returns a map that is the 'input_handle' with the given key-value pair inserted.
robots: noindex

# tf.raw_ops.TensorMapInsert

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>



Returns a map that is the 'input_handle' with the given key-value pair inserted.


<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.raw_ops.TensorMapInsert`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.raw_ops.TensorMapInsert(
    input_handle, key, value, name=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

input_handle: the original map
output_handle: the map with key and value inserted
key: the key to be inserted
value: the value to be inserted

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`input_handle`<a id="input_handle"></a>
</td>
<td>
A `Tensor` of type `variant`.
</td>
</tr><tr>
<td>
`key`<a id="key"></a>
</td>
<td>
A `Tensor`.
</td>
</tr><tr>
<td>
`value`<a id="value"></a>
</td>
<td>
A `Tensor`.
</td>
</tr><tr>
<td>
`name`<a id="name"></a>
</td>
<td>
A name for the operation (optional).
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A `Tensor` of type `variant`.
</td>
</tr>

</table>

