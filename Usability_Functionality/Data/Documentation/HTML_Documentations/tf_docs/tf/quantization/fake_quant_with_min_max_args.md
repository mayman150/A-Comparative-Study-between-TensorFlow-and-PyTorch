description: Fake-quantize the 'inputs' tensor, type float to 'outputs' tensor of same shape and type.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.quantization.fake_quant_with_min_max_args" />
<meta itemprop="path" content="Stable" />
</div>

# tf.quantization.fake_quant_with_min_max_args

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>



Fake-quantize the 'inputs' tensor, type float to 'outputs' tensor of same shape and type.


<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.fake_quant_with_min_max_args`, `tf.compat.v1.quantization.fake_quant_with_min_max_args`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.quantization.fake_quant_with_min_max_args(
    inputs: Annotated[Any, _atypes.Float32],
    min: float = -6,
    max: float = 6,
    num_bits: int = 8,
    narrow_range: bool = False,
    name=None
) -> Annotated[Any, _atypes.Float32]
</code></pre>



<!-- Placeholder for "Used in" -->


  Quantization is called fake since the output is still in floating point.
  The API converts inputs into values within the range [min and max] and returns
  as output.

Attributes

*   `[min; max]` define the clamping range for the `inputs` data.
*   `inputs` values are quantized into the quantization range (
`[0; 2^num_bits - 1]` when `narrow_range` is false and `[1; 2^num_bits - 1]`
when it is true) and then de-quantized and output as floats in `[min; max]`
interval.
*   `num_bits` is the bitwidth of the quantization; between 2 and 16, inclusive.

Before quantization, `min` and `max` values are adjusted with the following
logic.
It is suggested to have `min <= 0 <= max`. If `0` is not in the range of values,
the behavior can be unexpected:

*   If `0 < min < max`: `min_adj = 0` and `max_adj = max - min`.
*   If `min < max < 0`: `min_adj = min - max` and `max_adj = 0`.
*   If `min <= 0 <= max`: `scale = (max - min) / (2^num_bits - 1) `,
`min_adj = scale * round(min / scale)` and `max_adj = max + min_adj - min`.


Examples

```python

inp = tf.constant ([10.03, -10.23, 3])
out = tf.quantization.fake_quant_with_min_max_args(inp, min=-5, max=5,
                                                   num_bits=16)
print(out)

#  Output:
#  tf.Tensor([ 4.9999237 -5.0000763  3.0000763], shape=(3,), dtype=float32)
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Raises</h2></th></tr>
<tr class="alt">
<td colspan="2">
* InvalidArgumentError:
  - If num_bits are outside of range [2, 16].
  - If min >= max.
* ValueError: If `inputs` are of any other type than float32.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`inputs`<a id="inputs"></a>
</td>
<td>
A `Tensor` of type `float32`.
</td>
</tr><tr>
<td>
`min`<a id="min"></a>
</td>
<td>
An optional `float`. Defaults to `-6`.
</td>
</tr><tr>
<td>
`max`<a id="max"></a>
</td>
<td>
An optional `float`. Defaults to `6`.
</td>
</tr><tr>
<td>
`num_bits`<a id="num_bits"></a>
</td>
<td>
An optional `int`. Defaults to `8`.
</td>
</tr><tr>
<td>
`narrow_range`<a id="narrow_range"></a>
</td>
<td>
An optional `bool`. Defaults to `False`.
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
A `Tensor` of type `float32`.
</td>
</tr>

</table>

