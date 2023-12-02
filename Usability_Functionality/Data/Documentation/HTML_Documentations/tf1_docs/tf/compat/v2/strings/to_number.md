<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.compat.v2.strings.to_number" />
<meta itemprop="path" content="Stable" />
</div>

# tf.compat.v2.strings.to_number

Converts each string in the input Tensor to the specified numeric type.

``` python
tf.compat.v2.strings.to_number(
    input,
    out_type=tf.dtypes.float32,
    name=None
)
```

<!-- Placeholder for "Used in" -->

(Note that int32 overflow results in an error while float overflow
results in a rounded value.)

#### Args:


* <b>`input`</b>: A `Tensor` of type `string`.
* <b>`out_type`</b>: An optional <a href="../../../../tf/dtypes/DType.md"><code>tf.DType</code></a> from: `tf.float32, tf.float64, tf.int32,
  tf.int64`. Defaults to `tf.float32`.
  The numeric type to interpret each string in `string_tensor` as.
* <b>`name`</b>: A name for the operation (optional).


#### Returns:

A `Tensor` of type `out_type`.
