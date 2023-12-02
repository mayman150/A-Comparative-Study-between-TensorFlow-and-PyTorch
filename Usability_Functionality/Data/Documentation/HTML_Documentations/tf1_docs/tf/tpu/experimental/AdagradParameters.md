<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.tpu.experimental.AdagradParameters" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
</div>

# tf.tpu.experimental.AdagradParameters

## Class `AdagradParameters`

Optimization parameters for Adagrad with TPU embeddings.



### Aliases:

* Class `tf.compat.v1.tpu.experimental.AdagradParameters`
* Class `tf.compat.v2.compat.v1.tpu.experimental.AdagradParameters`
* Class `tf.tpu.experimental.AdagradParameters`

<!-- Placeholder for "Used in" -->

Pass this to <a href="../../../tf/estimator/tpu/experimental/EmbeddingConfigSpec.md"><code>tf.estimator.tpu.experimental.EmbeddingConfigSpec</code></a> via the
`optimization_parameters` argument to set the optimizer and its parameters.
See the documentation for <a href="../../../tf/estimator/tpu/experimental/EmbeddingConfigSpec.md"><code>tf.estimator.tpu.experimental.EmbeddingConfigSpec</code></a>
for more details.

```
estimator = tf.estimator.tpu.TPUEstimator(
    ...
    embedding_spec=tf.estimator.tpu.experimental.EmbeddingConfigSpec(
        ...
        optimization_parameters=tf.tpu.experimental.AdagradParameters(0.1),
        ...))
```

<h2 id="__init__"><code>__init__</code></h2>

``` python
__init__(
    learning_rate,
    initial_accumulator=0.1,
    use_gradient_accumulation=True,
    clip_weight_min=None,
    clip_weight_max=None
)
```

Optimization parameters for Adagrad.


#### Args:


* <b>`learning_rate`</b>: used for updating embedding table.
* <b>`initial_accumulator`</b>: initial accumulator for Adagrad.
* <b>`use_gradient_accumulation`</b>: setting this to `False` makes embedding
  gradients calculation less accurate but faster. Please see
  `optimization_parameters.proto` for details.
  for details.
* <b>`clip_weight_min`</b>: the minimum value to clip by; None means -infinity.
* <b>`clip_weight_max`</b>: the maximum value to clip by; None means +infinity.



