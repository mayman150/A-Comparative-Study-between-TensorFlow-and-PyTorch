description: Computes the sparse categorical crossentropy loss.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.keras.metrics.sparse_categorical_crossentropy" />
<meta itemprop="path" content="Stable" />
</div>

# tf.keras.metrics.sparse_categorical_crossentropy

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/losses.py#L2403-L2460">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Computes the sparse categorical crossentropy loss.


<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`tf.keras.losses.sparse_categorical_crossentropy`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.keras.metrics.sparse_categorical_crossentropy(
    y_true, y_pred, from_logits=False, axis=-1, ignore_class=None
)
</code></pre>



<!-- Placeholder for "Used in" -->


#### Standalone usage:



```
>>> y_true = [1, 2]
>>> y_pred = [[0.05, 0.95, 0], [0.1, 0.8, 0.1]]
>>> loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
>>> assert loss.shape == (2,)
>>> loss.numpy()
array([0.0513, 2.303], dtype=float32)
```

```
>>> y_true = [[[ 0,  2],
...            [-1, -1]],
...           [[ 0,  2],
...            [-1, -1]]]
>>> y_pred = [[[[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
...             [[0.2, 0.5, 0.3], [0.0, 1.0, 0.0]]],
...           [[[1.0, 0.0, 0.0], [0.0, 0.5, 0.5]],
...            [[0.2, 0.5, 0.3], [0.0, 1.0, 0.0]]]]
>>> loss = tf.keras.losses.sparse_categorical_crossentropy(
...   y_true, y_pred, ignore_class=-1)
>>> loss.numpy()
array([[[2.3841855e-07, 2.3841855e-07],
        [0.0000000e+00, 0.0000000e+00]],
       [[2.3841855e-07, 6.9314730e-01],
        [0.0000000e+00, 0.0000000e+00]]], dtype=float32)
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
Ground truth values.
</td>
</tr><tr>
<td>
`y_pred`<a id="y_pred"></a>
</td>
<td>
The predicted values.
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
`axis`<a id="axis"></a>
</td>
<td>
Defaults to -1. The dimension along which the entropy is
computed.
</td>
</tr><tr>
<td>
`ignore_class`<a id="ignore_class"></a>
</td>
<td>
Optional integer. The ID of a class to be ignored during
loss computation. This is useful, for example, in segmentation
problems featuring a "void" class (commonly -1 or 255) in
segmentation maps. By default (`ignore_class=None`), all classes are
considered.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
Sparse categorical crossentropy loss value.
</td>
</tr>

</table>

