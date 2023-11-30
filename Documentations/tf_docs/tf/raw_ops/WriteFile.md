description: Writes contents to the file at input filename.
robots: noindex

# tf.raw_ops.WriteFile

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>



Writes `contents` to the file at input `filename`.


<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.raw_ops.WriteFile`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.raw_ops.WriteFile(
    filename, contents, name=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

Creates the file and recursively creates directory if it does not exist.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`filename`<a id="filename"></a>
</td>
<td>
A `Tensor` of type `string`.
scalar. The name of the file to which we write the contents.
</td>
</tr><tr>
<td>
`contents`<a id="contents"></a>
</td>
<td>
A `Tensor` of type `string`.
scalar. The content to be written to the output file.
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
The created Operation.
</td>
</tr>

</table>

