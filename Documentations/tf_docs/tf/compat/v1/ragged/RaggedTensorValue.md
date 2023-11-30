description: Represents the value of a RaggedTensor.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.compat.v1.ragged.RaggedTensorValue" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="to_list"/>
</div>

# tf.compat.v1.ragged.RaggedTensorValue

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/ops/ragged/ragged_tensor_value.py">View source</a>



Represents the value of a `RaggedTensor`.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.compat.v1.ragged.RaggedTensorValue(
    values, row_splits
)
</code></pre>



<!-- Placeholder for "Used in" -->

Warning: `RaggedTensorValue` should only be used in graph mode; in
eager mode, the <a href="../../../../tf/RaggedTensor.md"><code>tf.RaggedTensor</code></a> class contains its value directly.

See <a href="../../../../tf/RaggedTensor.md"><code>tf.RaggedTensor</code></a> for a description of ragged tensors.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`values`<a id="values"></a>
</td>
<td>
A numpy array of any type and shape; or a RaggedTensorValue.
</td>
</tr><tr>
<td>
`row_splits`<a id="row_splits"></a>
</td>
<td>
A 1-D int32 or int64 numpy array.
</td>
</tr>
</table>





<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`dtype`<a id="dtype"></a>
</td>
<td>
The numpy dtype of values in this tensor.
</td>
</tr><tr>
<td>
`flat_values`<a id="flat_values"></a>
</td>
<td>
The innermost `values` array for this ragged tensor value.
</td>
</tr><tr>
<td>
`nested_row_splits`<a id="nested_row_splits"></a>
</td>
<td>
The row_splits for all ragged dimensions in this ragged tensor value.
</td>
</tr><tr>
<td>
`ragged_rank`<a id="ragged_rank"></a>
</td>
<td>
The number of ragged dimensions in this ragged tensor value.
</td>
</tr><tr>
<td>
`row_splits`<a id="row_splits"></a>
</td>
<td>
The split indices for the ragged tensor value.
</td>
</tr><tr>
<td>
`shape`<a id="shape"></a>
</td>
<td>
A tuple indicating the shape of this RaggedTensorValue.
</td>
</tr><tr>
<td>
`values`<a id="values"></a>
</td>
<td>
The concatenated values for all rows in this tensor.
</td>
</tr>
</table>



## Methods

<h3 id="to_list"><code>to_list</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/ops/ragged/ragged_tensor_value.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>to_list()
</code></pre>

Returns this ragged tensor value as a nested Python list.




