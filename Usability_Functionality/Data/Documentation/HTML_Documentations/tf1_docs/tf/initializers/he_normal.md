<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.initializers.he_normal" />
<meta itemprop="path" content="Stable" />
</div>

# tf.initializers.he_normal

He normal initializer.

### Aliases:

* `tf.compat.v1.initializers.he_normal`
* `tf.compat.v1.keras.initializers.he_normal`
* `tf.compat.v2.compat.v1.initializers.he_normal`
* `tf.compat.v2.compat.v1.keras.initializers.he_normal`
* `tf.initializers.he_normal`
* `tf.keras.initializers.he_normal`

``` python
tf.initializers.he_normal(seed=None)
```

<!-- Placeholder for "Used in" -->

It draws samples from a truncated normal distribution centered on 0
with standard deviation (after truncation) given by
`stddev = sqrt(2 / fan_in)` where `fan_in` is the number of
input units in the weight tensor.

#### Arguments:


* <b>`seed`</b>: A Python integer. Used to seed the random generator.


#### Returns:

An initializer.



#### References:

[He et al., 2015]
(https://www.cv-foundation.org/openaccess/content_iccv_2015/html/He_Delving_Deep_into_ICCV_2015_paper.html)
# pylint: disable=line-too-long
([pdf](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/He_Delving_Deep_into_ICCV_2015_paper.pdf))
