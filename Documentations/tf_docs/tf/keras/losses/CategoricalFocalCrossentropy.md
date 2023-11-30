description: Computes the alpha balanced focal crossentropy loss.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.keras.losses.CategoricalFocalCrossentropy" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__call__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="from_config"/>
<meta itemprop="property" content="get_config"/>
</div>

# tf.keras.losses.CategoricalFocalCrossentropy

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/losses.py#L933-L1070">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Computes the alpha balanced focal crossentropy loss.

Inherits From: [`Loss`](../../../tf/keras/losses/Loss.md)

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.keras.losses.CategoricalFocalCrossentropy(
    alpha=0.25,
    gamma=2.0,
    from_logits=False,
    label_smoothing=0.0,
    axis=-1,
    reduction=losses_utils.ReductionV2.AUTO,
    name=&#x27;categorical_focal_crossentropy&#x27;
)
</code></pre>



<!-- Placeholder for "Used in" -->

Use this crossentropy loss function when there are two or more label
classes and if you want to handle class imbalance without using
`class_weights`. We expect labels to be provided in a `one_hot`
representation.

According to [Lin et al., 2018](https://arxiv.org/pdf/1708.02002.pdf), it
helps to apply a focal factor to down-weight easy examples and focus more on
hard examples. The general formula for the focal loss (FL)
is as follows:

`FL(p_t) = (1 − p_t)^gamma * log(p_t)`

where `p_t` is defined as follows:
`p_t = output if y_true == 1, else 1 - output`

`(1 − p_t)^gamma` is the `modulating_factor`, where `gamma` is a focusing
parameter. When `gamma` = 0, there is no focal effect on the cross entropy.
`gamma` reduces the importance given to simple examples in a smooth manner.

The authors use alpha-balanced variant of focal loss (FL) in the paper:
`FL(p_t) = −alpha * (1 − p_t)^gamma * log(p_t)`

where `alpha` is the weight factor for the classes. If `alpha` = 1, the
loss won't be able to handle class imbalance properly as all
classes will have the same weight. This can be a constant or a list of
constants. If alpha is a list, it must have the same length as the number
of classes.

The formula above can be generalized to:
`FL(p_t) = alpha * (1 − p_t)^gamma * CrossEntropy(y_true, y_pred)`

where minus comes from `CrossEntropy(y_true, y_pred)` (CE).

Extending this to multi-class case is straightforward:
`FL(p_t) = alpha * (1 − p_t)^gamma * CategoricalCE(y_true, y_pred)`

In the snippet below, there is `# classes` floating pointing values per
example. The shape of both `y_pred` and `y_true` are
`[batch_size, num_classes]`.

#### Standalone usage:



```
>>> y_true = [[0., 1., 0.], [0., 0., 1.]]
>>> y_pred = [[0.05, 0.95, 0], [0.1, 0.8, 0.1]]
>>> # Using 'auto'/'sum_over_batch_size' reduction type.
>>> cce = tf.keras.losses.CategoricalFocalCrossentropy()
>>> cce(y_true, y_pred).numpy()
0.23315276
```

```
>>> # Calling with 'sample_weight'.
>>> cce(y_true, y_pred, sample_weight=tf.constant([0.3, 0.7])).numpy()
0.1632
```

```
>>> # Using 'sum' reduction type.
>>> cce = tf.keras.losses.CategoricalFocalCrossentropy(
...     reduction=tf.keras.losses.Reduction.SUM)
>>> cce(y_true, y_pred).numpy()
0.46631
```

```
>>> # Using 'none' reduction type.
>>> cce = tf.keras.losses.CategoricalFocalCrossentropy(
...     reduction=tf.keras.losses.Reduction.NONE)
>>> cce(y_true, y_pred).numpy()
array([3.2058331e-05, 4.6627346e-01], dtype=float32)
```

Usage with the `compile()` API:
```python
model.compile(optimizer='adam',
              loss=tf.keras.losses.CategoricalFocalCrossentropy())
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`alpha`<a id="alpha"></a>
</td>
<td>
A weight balancing factor for all classes, default is `0.25` as
mentioned in the reference. It can be a list of floats or a scalar.
In the multi-class case, alpha may be set by inverse class
frequency by using `compute_class_weight` from `sklearn.utils`.
</td>
</tr><tr>
<td>
`gamma`<a id="gamma"></a>
</td>
<td>
A focusing parameter, default is `2.0` as mentioned in the
reference. It helps to gradually reduce the importance given to
simple (easy) examples in a smooth manner.
</td>
</tr><tr>
<td>
`from_logits`<a id="from_logits"></a>
</td>
<td>
Whether `output` is expected to be a logits tensor. By
default, we consider that `output` encodes a probability
distribution.
</td>
</tr><tr>
<td>
`label_smoothing`<a id="label_smoothing"></a>
</td>
<td>
Float in [0, 1]. When > 0, label values are smoothed,
meaning the confidence on label values are relaxed. For example, if
`0.1`, use `0.1 / num_classes` for non-target labels and
`0.9 + 0.1 / num_classes` for target labels.
</td>
</tr><tr>
<td>
`axis`<a id="axis"></a>
</td>
<td>
The axis along which to compute crossentropy (the features
axis). Defaults to -1.
</td>
</tr><tr>
<td>
`reduction`<a id="reduction"></a>
</td>
<td>
Type of <a href="../../../tf/keras/losses/Reduction.md"><code>tf.keras.losses.Reduction</code></a> to apply to
loss. Default value is `AUTO`. `AUTO` indicates that the reduction
option will be determined by the usage context. For almost all cases
this defaults to `SUM_OVER_BATCH_SIZE`. When used under a
<a href="../../../tf/distribute/Strategy.md"><code>tf.distribute.Strategy</code></a>, except via <a href="../../../tf/keras/Model.md#compile"><code>Model.compile()</code></a> and
<a href="../../../tf/keras/Model.md#fit"><code>Model.fit()</code></a>, using `AUTO` or `SUM_OVER_BATCH_SIZE`
will raise an error. Please see this custom training [tutorial](
https://www.tensorflow.org/tutorials/distribute/custom_training)
for more details.
</td>
</tr><tr>
<td>
`name`<a id="name"></a>
</td>
<td>
Optional name for the instance.
Defaults to 'categorical_focal_crossentropy'.
</td>
</tr>
</table>



## Methods

<h3 id="from_config"><code>from_config</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/losses.py#L287-L301">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@classmethod</code>
<code>from_config(
    config
)
</code></pre>

Instantiates a `Loss` from its config (output of `get_config()`).


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`config`
</td>
<td>
Output of `get_config()`.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A <a href="../../../tf/keras/losses/Loss.md"><code>keras.losses.Loss</code></a> instance.
</td>
</tr>

</table>



<h3 id="get_config"><code>get_config</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/losses.py#L1064-L1070">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_config()
</code></pre>

Returns the config dictionary for a `Loss` instance.


<h3 id="__call__"><code>__call__</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/losses.py#L102-L163">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__call__(
    y_true, y_pred, sample_weight=None
)
</code></pre>

Invokes the `Loss` instance.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`y_true`
</td>
<td>
Ground truth values. shape = `[batch_size, d0, .. dN]`,
except sparse loss functions such as sparse categorical
crossentropy where shape = `[batch_size, d0, .. dN-1]`
</td>
</tr><tr>
<td>
`y_pred`
</td>
<td>
The predicted values. shape = `[batch_size, d0, .. dN]`
</td>
</tr><tr>
<td>
`sample_weight`
</td>
<td>
Optional `sample_weight` acts as a coefficient for
the loss. If a scalar is provided, then the loss is simply
scaled by the given value. If `sample_weight` is a tensor of
size `[batch_size]`, then the total loss for each sample of the
batch is rescaled by the corresponding element in the
`sample_weight` vector. If the shape of `sample_weight` is
`[batch_size, d0, .. dN-1]` (or can be broadcasted to this
shape), then each loss element of `y_pred` is scaled by the
corresponding value of `sample_weight`. (Note on`dN-1`: all loss
functions reduce by 1 dimension, usually axis=-1.)
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
Weighted loss float `Tensor`. If `reduction` is `NONE`, this has
shape `[batch_size, d0, .. dN-1]`; otherwise, it is scalar.
(Note `dN-1` because all loss functions reduce by 1 dimension,
usually axis=-1.)
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>

<tr>
<td>
`ValueError`
</td>
<td>
If the shape of `sample_weight` is invalid.
</td>
</tr>
</table>





