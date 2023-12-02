<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.tpu.experimental.StochasticGradientDescentParameters" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
</div>

# tf.tpu.experimental.StochasticGradientDescentParameters

## Class `StochasticGradientDescentParameters`

Optimization parameters for stochastic gradient descent for TPU embeddings.



### Aliases:

* Class `tf.compat.v1.tpu.experimental.StochasticGradientDescentParameters`
* Class `tf.compat.v2.compat.v1.tpu.experimental.StochasticGradientDescentParameters`
* Class `tf.tpu.experimental.StochasticGradientDescentParameters`

<!-- Placeholder for "Used in" -->

Pass this to <a href="../../../tf/estimator/tpu/experimental/EmbeddingConfigSpec.md"><code>tf.estimator.tpu.experimental.EmbeddingConfigSpec</code></a> via the
`optimization_parameters` argument to set the optimizer and its parameters.
See the documentation for <a href="../../../tf/estimator/tpu/experimental/EmbeddingConfigSpec.md"><code>tf.estimator.tpu.experimental.EmbeddingConfigSpec</code></a>
for more details.

```
estimator = tf.estimator.tpu.TPUEstimator(
    ...
    embedding_config_spec=tf.estimator.tpu.experimental.EmbeddingConfigSpec(
        ...
        optimization_parameters=(
            tf.tpu.experimental.StochasticGradientDescentParameters(0.1))))
```

<h2 id="__init__"><code>__init__</code></h2>

``` python
__init__(
    learning_rate,
    clip_weight_min=None,
    clip_weight_max=None
)
```

Optimization parameters for stochastic gradient descent.


#### Args:


* <b>`learning_rate`</b>: a floating point value. The learning rate.
* <b>`clip_weight_min`</b>: the minimum value to clip by; None means -infinity.
* <b>`clip_weight_max`</b>: the maximum value to clip by; None means +infinity.



