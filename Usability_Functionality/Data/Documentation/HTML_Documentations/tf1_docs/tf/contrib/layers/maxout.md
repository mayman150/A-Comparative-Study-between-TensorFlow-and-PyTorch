<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.contrib.layers.maxout" />
<meta itemprop="path" content="Stable" />
</div>

# tf.contrib.layers.maxout

Adds a maxout op from https://arxiv.org/abs/1302.4389

``` python
tf.contrib.layers.maxout(
    inputs,
    num_units,
    axis=-1,
    scope=None
)
```

<!-- Placeholder for "Used in" -->

"Maxout Networks" Ian J. Goodfellow, David Warde-Farley, Mehdi Mirza, Aaron
Courville,
 Yoshua Bengio

Usually the operation is performed in the filter/channel dimension. This can
also be
used after fully-connected layers to reduce number of features.

#### Arguments:


* <b>`inputs`</b>: Tensor input
* <b>`num_units`</b>: Specifies how many features will remain after maxout in the
  `axis` dimension (usually channel). This must be a factor of number of
  features.
* <b>`axis`</b>: The dimension where max pooling will be performed. Default is the last
  dimension.
* <b>`scope`</b>: Optional scope for variable_scope.


#### Returns:

A `Tensor` representing the results of the pooling operation.



#### Raises:


* <b>`ValueError`</b>: if num_units is not multiple of number of features.