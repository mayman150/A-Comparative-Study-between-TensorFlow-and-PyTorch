description: Computes the binary focal crossentropy loss.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.keras.metrics.binary_focal_crossentropy" />
<meta itemprop="path" content="Stable" />
</div>

# tf.keras.metrics.binary_focal_crossentropy

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/losses.py#L2573-L2658">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Computes the binary focal crossentropy loss.


<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`tf.keras.losses.binary_focal_crossentropy`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.keras.metrics.binary_focal_crossentropy(
    y_true,
    y_pred,
    apply_class_balancing=False,
    alpha=0.25,
    gamma=2.0,
    from_logits=False,
    label_smoothing=0.0,
    axis=-1
)
</code></pre>



<!-- Placeholder for "Used in" -->

According to [Lin et al., 2018](https://arxiv.org/pdf/1708.02002.pdf), it
helps to apply a focal factor to down-weight easy examples and focus more on
hard examples. By default, the focal tensor is computed as follows:

`focal_factor = (1 - output)**gamma` for class 1
`focal_factor = output**gamma` for class 0
where `gamma` is a focusing parameter. When `gamma` = 0, there is no focal
effect on the binary crossentropy loss.

If `apply_class_balancing == True`, this function also takes into account a
weight balancing factor for the binary classes 0 and 1 as follows:

`weight = alpha` for class 1 (`target == 1`)
`weight = 1 - alpha` for class 0
where `alpha` is a float in the range of `[0, 1]`.

#### Standalone usage:



```
>>> y_true = [[0, 1], [0, 0]]
>>> y_pred = [[0.6, 0.4], [0.4, 0.6]]
>>> loss = tf.keras.losses.binary_focal_crossentropy(y_true, y_pred,
...                                                  gamma=2)
>>> assert loss.shape == (2,)
>>> loss.numpy()
array([0.330, 0.206], dtype=float32)
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`y_true`<a id="y_true"></a>
</td>
<td>
Ground truth values, of shape `(batch_size, d0, .. dN)`.
</td>
</tr><tr>
<td>
`y_pred`<a id="y_pred"></a>
</td>
<td>
The predicted values, of shape `(batch_size, d0, .. dN)`.
</td>
</tr><tr>
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
mentioned in the reference. The weight for class 0 is `1.0 - alpha`.
</td>
</tr><tr>
<td>
`gamma`<a id="gamma"></a>
</td>
<td>
A focusing parameter, default is `2.0` as mentioned in the
reference.
</td>
</tr><tr>
<td>
`from_logits`<a id="from_logits"></a>
</td>
<td>
Whether `y_pred` is expected to be a logits tensor. By
default, we assume that `y_pred` encodes a probability distribution.
</td>
</tr><tr>
<td>
`label_smoothing`<a id="label_smoothing"></a>
</td>
<td>
Float in `[0, 1]`. If higher than 0 then smooth the
labels by squeezing them towards `0.5`, i.e., using `1. - 0.5 *
label_smoothing` for the target class and `0.5 * label_smoothing`
for the non-target class.
</td>
</tr><tr>
<td>
`axis`<a id="axis"></a>
</td>
<td>
The axis along which the mean is computed. Defaults to `-1`.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
Binary focal crossentropy loss value.
shape = `[batch_size, d0, .. dN-1]`.
</td>
</tr>

</table>

