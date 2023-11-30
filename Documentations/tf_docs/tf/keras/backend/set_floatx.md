description: Sets the default float type.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.keras.backend.set_floatx" />
<meta itemprop="path" content="Stable" />
</div>

# tf.keras.backend.set_floatx

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/backend_config.py#L82-L114">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Sets the default float type.


<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.keras.backend.set_floatx(
    value
)
</code></pre>



<!-- Placeholder for "Used in" -->

Note: It is not recommended to set this to float16 for training, as this
will likely cause numeric stability issues. Instead, mixed precision, which
is using a mix of float16 and float32, can be used by calling
`tf.keras.mixed_precision.set_global_policy('mixed_float16')`. See the
[mixed precision guide](
  https://www.tensorflow.org/guide/keras/mixed_precision) for details.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`value`<a id="value"></a>
</td>
<td>
String; `'float16'`, `'float32'`, or `'float64'`.
</td>
</tr>
</table>



#### Example:


```
>>> tf.keras.backend.floatx()
'float32'
>>> tf.keras.backend.set_floatx('float64')
>>> tf.keras.backend.floatx()
'float64'
>>> tf.keras.backend.set_floatx('float32')
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Raises</h2></th></tr>

<tr>
<td>
`ValueError`<a id="ValueError"></a>
</td>
<td>
In case of invalid value.
</td>
</tr>
</table>

