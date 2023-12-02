<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.nn.swish" />
<meta itemprop="path" content="Stable" />
</div>

# tf.nn.swish

Computes the Swish activation function: `x * sigmoid(x)`.

### Aliases:

* `tf.compat.v1.nn.swish`
* `tf.compat.v2.compat.v1.nn.swish`
* `tf.compat.v2.nn.swish`
* `tf.nn.swish`

``` python
tf.nn.swish(features)
```

<!-- Placeholder for "Used in" -->

Source: "Searching for Activation Functions" (Ramachandran et al. 2017)
https://arxiv.org/abs/1710.05941

#### Args:


* <b>`features`</b>: A `Tensor` representing preactivation values.
* <b>`name`</b>: A name for the operation (optional).


#### Returns:

The activation value.
