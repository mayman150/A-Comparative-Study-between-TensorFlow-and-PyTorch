description: Computes the maximum along segments of a tensor.
robots: noindex

# tf.raw_ops.SegmentMaxV2

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>



Computes the maximum along segments of a tensor.


<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.raw_ops.SegmentMaxV2`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.raw_ops.SegmentMaxV2(
    data, segment_ids, num_segments, name=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

Read
[the section on segmentation](https://tensorflow.org/api_docs/python/tf/math#Segmentation)
for an explanation of segments.

Computes a tensor such that
\\(output_i = \max_j(data_j)\\) where `max` is over `j` such
that `segment_ids[j] == i`.

If the maximum is empty for a given segment ID `i`, it outputs the smallest
possible value for the specific numeric type,
`output[i] = numeric_limits<T>::lowest()`.

Note: That this op is currently only supported with jit_compile=True.

Caution: On CPU, values in `segment_ids` are always validated to be sorted,
and an error is thrown for indices that are not increasing. On GPU, this
does not throw an error for unsorted indices. On GPU, out-of-order indices
result in safe but unspecified behavior, which may include treating
out-of-order indices as the same as a smaller following index.

The only difference with SegmentMax is the additional input  `num_segments`.
This helps in evaluating the output shape in compile time.
`num_segments` should be consistent with segment_ids.
e.g. Max(segment_ids) should be equal to `num_segments` - 1 for a 1-d segment_ids
With inconsistent num_segments, the op still runs. only difference is,
the output takes the size of num_segments irrespective of size of segment_ids and data.
for num_segments less than expected output size, the last elements are ignored
for num_segments more than the expected output size, last elements are assigned 
smallest possible value for the specific numeric type.

#### For example:



```
>>> @tf.function(jit_compile=True)
... def test(c):
...   return tf.raw_ops.SegmentMaxV2(data=c, segment_ids=tf.constant([0, 0, 1]), num_segments=2)
>>> c = tf.constant([[1,2,3,4], [4, 3, 2, 1], [5,6,7,8]])
>>> test(c).numpy()
array([[4, 3, 3, 4],
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
A `Tensor`. Must be one of the following types: `float32`, `float64`, `int32`, `uint8`, `int16`, `int8`, `int64`, `bfloat16`, `uint16`, `half`, `uint32`, `uint64`.
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

