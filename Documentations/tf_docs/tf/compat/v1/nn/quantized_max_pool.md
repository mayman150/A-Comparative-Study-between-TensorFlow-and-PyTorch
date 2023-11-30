description: Produces the max pool of the input tensor for quantized types.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.compat.v1.nn.quantized_max_pool" />
<meta itemprop="path" content="Stable" />
</div>

# tf.compat.v1.nn.quantized_max_pool

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>



Produces the max pool of the input tensor for quantized types.


<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.compat.v1.nn.quantized_max_pool(
    input: Annotated[Any, TV_QuantizedMaxPool_T],
    min_input: Annotated[Any, _atypes.Float32],
    max_input: Annotated[Any, _atypes.Float32],
    ksize,
    strides,
    padding: str,
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
`input`<a id="input"></a>
</td>
<td>
A `Tensor`. Must be one of the following types: `qint8`, `quint8`, `qint32`, `qint16`, `quint16`.
The 4D (batch x rows x cols x depth) Tensor to MaxReduce over.
</td>
</tr><tr>
<td>
`min_input`<a id="min_input"></a>
</td>
<td>
A `Tensor` of type `float32`.
The float value that the lowest quantized input value represents.
</td>
</tr><tr>
<td>
`max_input`<a id="max_input"></a>
</td>
<td>
A `Tensor` of type `float32`.
The float value that the highest quantized input value represents.
</td>
</tr><tr>
<td>
`ksize`<a id="ksize"></a>
</td>
<td>
A list of `ints`.
The size of the window for each dimension of the input tensor.
The length must be 4 to match the number of dimensions of the input.
</td>
</tr><tr>
<td>
`strides`<a id="strides"></a>
</td>
<td>
A list of `ints`.
The stride of the sliding window for each dimension of the input
tensor. The length must be 4 to match the number of dimensions of the input.
</td>
</tr><tr>
<td>
`padding`<a id="padding"></a>
</td>
<td>
A `string` from: `"SAME", "VALID"`.
The type of padding algorithm to use.
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
A tuple of `Tensor` objects (output, min_output, max_output).
</td>
</tr>
<tr>
<td>
`output`<a id="output"></a>
</td>
<td>
A `Tensor`. Has the same type as `input`.
</td>
</tr><tr>
<td>
`min_output`<a id="min_output"></a>
</td>
<td>
A `Tensor` of type `float32`.
</td>
</tr><tr>
<td>
`max_output`<a id="max_output"></a>
</td>
<td>
A `Tensor` of type `float32`.
</td>
</tr>
</table>

