description: Computes focal cross-entropy loss between true labels and predictions.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.keras.losses.BinaryFocalCrossentropy" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__call__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="from_config"/>
<meta itemprop="property" content="get_config"/>
</div>

# tf.keras.losses.BinaryFocalCrossentropy

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/losses.py#L670-L841">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Computes focal cross-entropy loss between true labels and predictions.

Inherits From: [`Loss`](../../../tf/keras/losses/Loss.md)

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.keras.losses.BinaryFocalCrossentropy(
    apply_class_balancing=False,
    alpha=0.25,
    gamma=2.0,
    from_logits=False,
    label_smoothing=0.0,
    axis=-1,
    reduction=losses_utils.ReductionV2.AUTO,
    name=&#x27;binary_focal_crossentropy&#x27;
)
</code></pre>



<!-- Placeholder for "Used in" -->

Binary cross-entropy loss is often used for binary (0 or 1) classification
tasks. The loss function requires the following inputs:

- `y_true` (true label): This is either 0 or 1.
- `y_pred` (predicted value): This is the model's prediction, i.e, a single
    floating-point value which either represents a
    [logit](https://en.wikipedia.org/wiki/Logit), (i.e, value in [-inf, inf]
    when `from_logits=True`) or a probability (i.e, value in [0., 1.] when
    `from_logits=False`).

According to [Lin et al., 2018](https://arxiv.org/pdf/1708.02002.pdf), it
helps to apply a "focal factor" to down-weight easy examples and focus more
on hard examples. By default, the focal tensor is computed as follows:

`focal_factor = (1 - output) ** gamma` for class 1
`focal_factor = output ** gamma` for class 0
where `gamma` is a focusing parameter. When `gamma=0`, this function is
equivalent to the binary crossentropy loss.

With the `compile()` API:

```python
model.compile(
  loss=tf.keras.losses.BinaryFocalCrossentropy(gamma=2.0, from_logits=True),
  ....
)
```

As a standalone function:

```
>>> # Example 1: (batch_size = 1, number of samples = 4)
>>> y_true = [0, 1, 0, 0]
>>> y_pred = [-18.6, 0.51, 2.94, -12.8]
>>> loss = tf.keras.losses.BinaryFocalCrossentropy(gamma=2,
...                                                from_logits=True)
>>> loss(y_true, y_pred).numpy()
0.691
```

```
>>> # Apply class weight
>>> loss = tf.keras.losses.BinaryFocalCrossentropy(
...     apply_class_balancing=True, gamma=2, from_logits=True)
>>> loss(y_true, y_pred).numpy()
0.51
```

```
>>> # Example 2: (batch_size = 2, number of samples = 4)
>>> y_true = [[0, 1], [0, 0]]
>>> y_pred = [[-18.6, 0.51], [2.94, -12.8]]
>>> # Using default 'auto'/'sum_over_batch_size' reduction type.
>>> loss = tf.keras.losses.BinaryFocalCrossentropy(gamma=3,
...                                                from_logits=True)
>>> loss(y_true, y_pred).numpy()
0.647
```

```
>>> # Apply class weight
>>> loss = tf.keras.losses.BinaryFocalCrossentropy(
...     apply_class_balancing=True, gamma=3, from_logits=True)
>>> loss(y_true, y_pred).numpy()
0.482
```

```
>>> # Using 'sample_weight' attribute with focal effect
>>> loss = tf.keras.losses.BinaryFocalCrossentropy(gamma=3,
...                                                from_logits=True)
>>> loss(y_true, y_pred, sample_weight=[0.8, 0.2]).numpy()
0.133
```

```
>>> # Apply class weight
>>> loss = tf.keras.losses.BinaryFocalCrossentropy(
...     apply_class_balancing=True, gamma=3, from_logits=True)
>>> loss(y_true, y_pred, sample_weight=[0.8, 0.2]).numpy()
0.097
```

```
>>> # Using 'sum' reduction` type.
>>> loss = tf.keras.losses.BinaryFocalCrossentropy(gamma=4,
...                                                from_logits=True,
...     reduction=tf.keras.losses.Reduction.SUM)
>>> loss(y_true, y_pred).numpy()
1.222
```

```
>>> # Apply class weight
>>> loss = tf.keras.losses.BinaryFocalCrossentropy(
...     apply_class_balancing=True, gamma=4, from_logits=True,
...     reduction=tf.keras.losses.Reduction.SUM)
>>> loss(y_true, y_pred).numpy()
0.914
```

```
>>> # Using 'none' reduction type.
>>> loss = tf.keras.losses.BinaryFocalCrossentropy(
...     gamma=5, from_logits=True,
...     reduction=tf.keras.losses.Reduction.NONE)
>>> loss(y_true, y_pred).numpy()
array([0.0017 1.1561], dtype=float32)
```

```
>>> # Apply class weight
>>> loss = tf.keras.losses.BinaryFocalCrossentropy(
...     apply_class_balancing=True, gamma=5, from_logits=True,
...     reduction=tf.keras.losses.Reduction.NONE)
>>> loss(y_true, y_pred).numpy()
array([0.0004 0.8670], dtype=float32)
```


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`apply_class_balancing`<a id="apply_class_balancing"></a>
</td>
<td>
A bool, whether to apply weight balancing on the
binary classes 0 and 1.
</td>
</tr><tr>
<td>
`alpha`<a id="alpha"></a>
</td>
<td>
A weight balancing factor for class 1, default is `0.25` as
mentioned in reference [Lin et al., 2018](
https://arxiv.org/pdf/1708.02002.pdf).  The weight for class 0 is
`1.0 - alpha`.
</td>
</tr><tr>
<td>
`gamma`<a id="gamma"></a>
</td>
<td>
A focusing parameter used to compute the focal factor, default is
`2.0` as mentioned in the reference
[Lin et al., 2018](https://arxiv.org/pdf/1708.02002.pdf).
</td>
</tr><tr>
<td>
`from_logits`<a id="from_logits"></a>
</td>
<td>
Whether to interpret `y_pred` as a tensor of
[logit](https://en.wikipedia.org/wiki/Logit) values. By default, we
assume that `y_pred` are probabilities (i.e., values in `[0, 1]`).
</td>
</tr><tr>
<td>
`label_smoothing`<a id="label_smoothing"></a>
</td>
<td>
Float in `[0, 1]`. When `0`, no smoothing occurs.
When > `0`, we compute the loss between the predicted labels and a
smoothed version of the true labels, where the smoothing squeezes
the labels towards `0.5`. Larger values of `label_smoothing`
correspond to heavier smoothing.
</td>
</tr><tr>
<td>
`axis`<a id="axis"></a>
</td>
<td>
The axis along which to compute crossentropy (the features axis).
Defaults to `-1`.
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
Name for the op. Defaults to 'binary_focal_crossentropy'.
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

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/losses.py#L834-L841">View source</a>

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





