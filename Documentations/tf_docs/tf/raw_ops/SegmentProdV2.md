description: Computes the product along segments of a tensor.
robots: noindex

# tf.raw_ops.SegmentProdV2

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>



Computes the product along segments of a tensor.


<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.raw_ops.SegmentProdV2`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.raw_ops.SegmentProdV2(
    data, segment_ids, num_segments, name=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

Read
[the section on segmentation](https://tensorflow.org/api_docs/python/tf/math#Segmentation)
for an explanation of segments.

Computes a tensor such that
\\(output_i = \prod_j data_j\\) where the product is over `j` such
that `segment_ids[j] == i`.

If the product is empty for a given segment ID `i`, `output[i] = 1`.

Note: That this op is currently only supported with jit_compile=True.

The only difference with SegmentProd is the additional input  `num_segments`.
This helps in evaluating the output shape in compile time.
`num_segments` should be consistent with segment_ids.
e.g. Max(segment_ids) - 1 should be equal to `num_segments` for a 1-d segment_ids
With inconsistent num_segments, the op still runs. only difference is, 
the output takes the size of num_segments irrespective of size of segment_ids and data.
for num_segments less than expected output size, the last elements are ignored
for num_segments more than the expected output size, last elements are assigned 1.

#### For example:



```
>>> @tf.function(jit_compile=True)
... def test(c):
...   return tf.raw_ops.SegmentProdV2(data=c, segment_ids=tf.constant([0, 0, 1]), num_segments=2)
>>> c = tf.constant([[1,2,3,4], [4, 3, 2, 1], [5,6,7,8]])
>>> test(c).numpy()
array([[4, 6, 6, 4],
       [5, 6, 7, 8]], dtype=int32)
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`data`<a id="data"></a>
</td>
<td>
A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, `int64`, `qint8`, `quint8`, `qint32`, `bfloat16`, `qint16`, `quint16`, `uint16`, `complex128`, `half`, `uint32`, `uint64`.
</td>
</tr><tr>
<td>
`segment_ids`<a id="segment_ids"></a>
</td>
<td>
A `Tensor`. Must be one of the following types: `int32`, `int64`.
A 1-D tensor whose size is equal to the size of `data`'s
first dimension.  Values should be sorted and can be repeated.
The values must be less than `num_segments`.

Caution: The values are always validated to be sorted on CPU, never validated
on GPU.
</td>
</tr><tr>
<td>
`num_segments`<a id="num_segments"></a>
</td>
<td>
A `Tensor`. Must be one of the following types: `int32`, `int64`.
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
A `Tensor`. Has the same type as `data`.
</td>
</tr>

</table>

