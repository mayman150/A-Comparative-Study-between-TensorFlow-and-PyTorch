description: ND inverse fast Fourier transform.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.ifftnd" />
<meta itemprop="path" content="Stable" />
</div>

# tf.ifftnd

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>



ND inverse fast Fourier transform.


<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.ifftnd`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.ifftnd(
    input: Annotated[Any, TV_IFFTND_Tcomplex],
    fft_length: Annotated[Any, _atypes.Int32],
    axes: Annotated[Any, _atypes.Int32],
    name=None
) -> Annotated[Any, TV_IFFTND_Tcomplex]
</code></pre>



<!-- Placeholder for "Used in" -->

Computes the n-dimensional inverse discrete Fourier transform over designated
dimensions of `input`. The designated dimensions of `input` are assumed to be
the result of `IFFTND`.

If fft_length[i]<shape(input)[i], the input is cropped. If
fft_length[i]>shape(input)[i], the input is padded with zeros. If fft_length
is not given, the default shape(input) is used.

Axes mean the dimensions to perform the transform on. Default is to perform on
all axes.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`input`<a id="input"></a>
</td>
<td>
A `Tensor`. Must be one of the following types: `complex64`, `complex128`.
A complex tensor.
</td>
</tr><tr>
<td>
`fft_length`<a id="fft_length"></a>
</td>
<td>
A `Tensor` of type `int32`.
An int32 tensor. The FFT length for each dimension.
</td>
</tr><tr>
<td>
`axes`<a id="axes"></a>
</td>
<td>
A `Tensor` of type `int32`.
An int32 tensor with a same shape as fft_length. Axes to perform the transform.
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
A `Tensor`. Has the same type as `input`.
</td>
</tr>

</table>

