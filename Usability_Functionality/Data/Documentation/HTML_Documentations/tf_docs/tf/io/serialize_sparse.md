description: Serialize a SparseTensor into a 3-vector (1-D Tensor) object.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.io.serialize_sparse" />
<meta itemprop="path" content="Stable" />
</div>

# tf.io.serialize_sparse

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/ops/sparse_ops.py">View source</a>



Serialize a `SparseTensor` into a 3-vector (1-D `Tensor`) object.


<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.io.serialize_sparse(
    sp_input,
    out_type=<a href="../../tf/dtypes.md#string"><code>tf.dtypes.string</code></a>,
    name=None
)
</code></pre>



<!-- Placeholder for "Used in" -->


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`sp_input`<a id="sp_input"></a>
</td>
<td>
The input `SparseTensor`.
</td>
</tr><tr>
<td>
`out_type`<a id="out_type"></a>
</td>
<td>
The `dtype` to use for serialization.
</td>
</tr><tr>
<td>
`name`<a id="name"></a>
</td>
<td>
A name prefix for the returned tensors (optional).
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A 3-vector (1-D `Tensor`), with each column representing the serialized
`SparseTensor`'s indices, values, and shape (respectively).
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Raises</h2></th></tr>

<tr>
<td>
`TypeError`<a id="TypeError"></a>
</td>
<td>
If `sp_input` is not a `SparseTensor`.
</td>
</tr>
</table>

