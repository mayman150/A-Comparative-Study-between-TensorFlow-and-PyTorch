description: Zero-padding layer for 1D input (e.g. temporal sequence).

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.keras.layers.ZeroPadding1D" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
</div>

# tf.keras.layers.ZeroPadding1D

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/layers/reshaping/zero_padding1d.py#L29-L94">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Zero-padding layer for 1D input (e.g. temporal sequence).

Inherits From: [`Layer`](../../../tf/keras/layers/Layer.md), [`Module`](../../../tf/Module.md)

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.keras.layers.ZeroPadding1D(
    padding=1, **kwargs
)
</code></pre>



<!-- Placeholder for "Used in" -->


#### Examples:



```
>>> input_shape = (2, 2, 3)
>>> x = np.arange(np.prod(input_shape)).reshape(input_shape)
>>> print(x)
[[[ 0  1  2]
  [ 3  4  5]]
 [[ 6  7  8]
  [ 9 10 11]]]
>>> y = tf.keras.layers.ZeroPadding1D(padding=2)(x)
>>> print(y)
tf.Tensor(
  [[[ 0  0  0]
    [ 0  0  0]
    [ 0  1  2]
    [ 3  4  5]
    [ 0  0  0]
    [ 0  0  0]]
   [[ 0  0  0]
    [ 0  0  0]
    [ 6  7  8]
    [ 9 10 11]
    [ 0  0  0]
    [ 0  0  0]]], shape=(2, 6, 3), dtype=int64)
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`padding`<a id="padding"></a>
</td>
<td>
Int, or tuple of int (length 2).
- If int:
How many zeros to add at the beginning and end of
the padding dimension (axis 1).
- If tuple of int (length 2):
How many zeros to add at the beginning and the end of
the padding dimension (`(left_pad, right_pad)`).
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Input shape</h2></th></tr>
<tr class="alt">
<td colspan="2">
3D tensor with shape `(batch_size, axis_to_pad, features)`
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Output shape</h2></th></tr>
<tr class="alt">
<td colspan="2">
3D tensor with shape `(batch_size, padded_axis, features)`
</td>
</tr>

</table>



