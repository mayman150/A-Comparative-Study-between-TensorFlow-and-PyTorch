<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.contrib.learn.infer_real_valued_columns_from_input" />
<meta itemprop="path" content="Stable" />
</div>

# tf.contrib.learn.infer_real_valued_columns_from_input

Creates `FeatureColumn` objects for inputs defined by input `x`. (deprecated)

``` python
tf.contrib.learn.infer_real_valued_columns_from_input(x)
```

<!-- Placeholder for "Used in" -->

Warning: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating:
Please specify feature columns explicitly.

This interprets all inputs as dense, fixed-length float values.

#### Args:


* <b>`x`</b>: Real-valued matrix of shape [n_samples, n_features...]. Can be
   iterator that returns arrays of features.


#### Returns:

List of `FeatureColumn` objects.
