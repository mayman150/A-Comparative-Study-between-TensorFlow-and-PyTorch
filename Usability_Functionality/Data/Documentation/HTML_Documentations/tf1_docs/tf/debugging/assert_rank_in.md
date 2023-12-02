<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.debugging.assert_rank_in" />
<meta itemprop="path" content="Stable" />
</div>

# tf.debugging.assert_rank_in

Assert `x` has rank in `ranks`.

### Aliases:

* `tf.assert_rank_in`
* `tf.compat.v1.assert_rank_in`
* `tf.compat.v1.debugging.assert_rank_in`
* `tf.compat.v2.compat.v1.assert_rank_in`
* `tf.compat.v2.compat.v1.debugging.assert_rank_in`
* `tf.debugging.assert_rank_in`

``` python
tf.debugging.assert_rank_in(
    x,
    ranks,
    data=None,
    summarize=None,
    message=None,
    name=None
)
```

<!-- Placeholder for "Used in" -->

Example of adding a dependency to an operation:

```python
with tf.control_dependencies([tf.compat.v1.assert_rank_in(x, (2, 4))]):
  output = tf.reduce_sum(x)
```

#### Args:


* <b>`x`</b>:  Numeric `Tensor`.
* <b>`ranks`</b>:  Iterable of scalar `Tensor` objects.
* <b>`data`</b>:  The tensors to print out if the condition is False.  Defaults to
  error message and first few entries of `x`.
* <b>`summarize`</b>: Print this many entries of each tensor.
* <b>`message`</b>: A string to prefix to the default message.
* <b>`name`</b>: A name for this operation (optional).
  Defaults to "assert_rank_in".


#### Returns:

Op raising `InvalidArgumentError` unless rank of `x` is in `ranks`.
If static checks determine `x` has matching rank, a `no_op` is returned.



#### Raises:


* <b>`ValueError`</b>:  If static checks determine `x` has mismatched rank.