description: Computes the binary crossentropy loss.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.keras.metrics.binary_crossentropy" />
<meta itemprop="path" content="Stable" />
</div>

# tf.keras.metrics.binary_crossentropy

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/losses.py#L2488-L2534">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Computes the binary crossentropy loss.


<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`tf.keras.losses.binary_crossentropy`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.keras.metrics.binary_crossentropy(
    y_true, y_pred, from_logits=False, label_smoothing=0.0, axis=-1
)
</code></pre>



<!-- Placeholder for "Used in" -->


#### Standalone usage:



```
>>> y_true = [[0, 1], [0, 0]]
>>> y_pred = [[0.6, 0.4], [0.4, 0.6]]
>>> loss = tf.keras.losses.binary_crossentropy(y_true, y_pred)
>>> assert loss.shape == (2,)
>>> loss.numpy()
array([0.916 , 0.714], dtype=float32)
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
Ground truth values. shape = `[batch_size, d0, .. dN]`.
</td>
</tr><tr>
<td>
`y_pred`<a id="y_pred"></a>
</td>
<td>
The predicted values. shape = `[batch_size, d0, .. dN]`.
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
Float in [0, 1]. If > `0` then smooth the labels by
squeezing them towards 0.5 That is, using
`1. - 0.5 * label_smoothing` for the target class and
`0.5 * label_smoothing` for the non-target class.
</td>
</tr><tr>
<td>
`axis`<a id="axis"></a>
</td>
<td>
The axis along which the mean is computed. Defaults to -1.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
Binary crossentropy loss value. shape = `[batch_size, d0, .. dN-1]`.
</td>
</tr>

</table>

