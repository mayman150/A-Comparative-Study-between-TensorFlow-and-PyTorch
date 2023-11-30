description: Permute input tensor from src_format to dst_format.
robots: noindex

# tf.raw_ops.DataFormatVecPermute

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>



Permute input tensor from `src_format` to `dst_format`.


<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.raw_ops.DataFormatVecPermute`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.raw_ops.DataFormatVecPermute(
    x, src_format=&#x27;NHWC&#x27;, dst_format=&#x27;NCHW&#x27;, name=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

Given source and destination format strings of length n=4 or 5, the input
tensor must be a vector of size n or n-2, or a 2D tensor of shape
(n, 2) or (n-2, 2).

If the first dimension of the input tensor is n-2, it is assumed that
non-spatial dimensions are omitted (i.e `N`, `C`).

For example, with `src_format` of `NHWC`, `dst_format` of `NCHW`, and input:
```
[1, 2, 3, 4]
```
, the output will be:
```
[1, 4, 2, 3]
```
With `src_format` of `NDHWC`, `dst_format` of `NCDHW`, and input:
```
[[1, 6], [2, 7], [3, 8], [4, 9], [5, 10]]
```
, the output will be:
```
[[1, 6], [5, 10], [2, 7], [3, 8], [4, 9]]
```
With `src_format` of `NHWC`, `dst_format` of `NCHW`, and input:
```
[1, 2]
```
, the output will be:
```
[1, 2]
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`x`<a id="x"></a>
</td>
<td>
A `Tensor`. Must be one of the following types: `int32`, `int64`.
Tensor of rank 1 or 2 in source data format.
</td>
</tr><tr>
<td>
`src_format`<a id="src_format"></a>
</td>
<td>
An optional `string`. Defaults to `"NHWC"`.
source data format.
</td>
</tr><tr>
<td>
`dst_format`<a id="dst_format"></a>
</td>
<td>
An optional `string`. Defaults to `"NCHW"`.
destination data format.
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
A `Tensor`. Has the same type as `x`.
</td>
</tr>

</table>

