description: Computes the mean absolute percentage error between y_true & y_pred.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.keras.metrics.mean_absolute_percentage_error" />
<meta itemprop="path" content="Stable" />
</div>

# tf.keras.metrics.mean_absolute_percentage_error

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/losses.py#L1857-L1895">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Computes the mean absolute percentage error between `y_true` & `y_pred`.


<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`tf.keras.losses.MAPE`, `tf.keras.losses.mape`, `tf.keras.losses.mean_absolute_percentage_error`, `tf.keras.metrics.MAPE`, `tf.keras.metrics.mape`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.keras.metrics.mean_absolute_percentage_error(
    y_true, y_pred
)
</code></pre>



<!-- Placeholder for "Used in" -->

`loss = 100 * mean(abs((y_true - y_pred) / y_true), axis=-1)`

#### Standalone usage:



```
>>> y_true = np.random.random(size=(2, 3))
>>> y_true = np.maximum(y_true, 1e-7)  # Prevent division by zero
>>> y_pred = np.random.random(size=(2, 3))
>>> loss = tf.keras.losses.mean_absolute_percentage_error(y_true, y_pred)
>>> assert loss.shape == (2,)
>>> assert np.array_equal(
...     loss.numpy(),
...     100. * np.mean(np.abs((y_true - y_pred) / y_true), axis=-1))
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
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
Mean absolute percentage error values. shape = `[batch_size, d0, ..
dN-1]`.
</td>
</tr>

</table>

