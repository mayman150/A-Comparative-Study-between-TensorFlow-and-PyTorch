<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.contrib.data.rejection_resample" />
<meta itemprop="path" content="Stable" />
</div>

# tf.contrib.data.rejection_resample

A transformation that resamples a dataset to achieve a target distribution. (deprecated)

``` python
tf.contrib.data.rejection_resample(
    class_func,
    target_dist,
    initial_dist=None,
    seed=None
)
```

<!-- Placeholder for "Used in" -->

Warning: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating:
Use <a href="../../../tf/data/experimental/rejection_resample.md"><code>tf.data.experimental.rejection_resample(...)</code></a>.

**NOTE** Resampling is performed via rejection sampling; some fraction
of the input values will be dropped.

#### Args:


* <b>`class_func`</b>: A function mapping an element of the input dataset to a scalar
  <a href="../../../tf.md#int32"><code>tf.int32</code></a> tensor. Values should be in `[0, num_classes)`.
* <b>`target_dist`</b>: A floating point type tensor, shaped `[num_classes]`.
* <b>`initial_dist`</b>: (Optional.)  A floating point type tensor, shaped
  `[num_classes]`.  If not provided, the true class distribution is
  estimated live in a streaming fashion.
* <b>`seed`</b>: (Optional.) Python integer seed for the resampler.


#### Returns:

A `Dataset` transformation function, which can be passed to
<a href="../../../tf/data/Dataset.md#apply"><code>tf.data.Dataset.apply</code></a>.
