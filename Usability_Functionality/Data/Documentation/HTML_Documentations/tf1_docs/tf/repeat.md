<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.repeat" />
<meta itemprop="path" content="Stable" />
</div>

# tf.repeat

Repeat elements of `input`

### Aliases:

* `tf.compat.v1.repeat`
* `tf.compat.v2.compat.v1.repeat`
* `tf.compat.v2.repeat`
* `tf.repeat`

``` python
tf.repeat(
    input,
    repeats,
    axis=None,
    name=None
)
```

<!-- Placeholder for "Used in" -->


#### Args:


* <b>`input`</b>: An `N`-dimensional Tensor.
* <b>`repeats`</b>: An 1-D `int` Tensor. The number of repetitions for each element.
  repeats is broadcasted to fit the shape of the given axis. `len(repeats)`
  must equal `input.shape[axis]` if axis is not None.
* <b>`axis`</b>: An int. The axis along which to repeat values. By default (axis=None),
  use the flattened input array, and return a flat output array.
* <b>`name`</b>: A name for the operation.


#### Returns:

A Tensor which has the same shape as `input`, except along the given axis.
  If axis is None then the output array is flattened to match the flattened
  input array.

#### Examples:
  ```python
  >>> repeat(['a', 'b', 'c'], repeats=[3, 0, 2], axis=0)
  ['a', 'a', 'a', 'c', 'c']
  >>> repeat([[1, 2], [3, 4]], repeats=[2, 3], axis=0)
  [[1, 2], [1, 2], [3, 4], [3, 4], [3, 4]]
  >>> repeat([[1, 2], [3, 4]], repeats=[2, 3], axis=1)
  [[1, 1, 2, 2, 2], [3, 3, 4, 4, 4]]
  >>> repeat(3, repeats=4)
  [3, 3, 3, 3]
  >>> repeat([[1,2], [3,4]], repeats=2)
  [1, 1, 2, 2, 3, 3, 4, 4]
  ```