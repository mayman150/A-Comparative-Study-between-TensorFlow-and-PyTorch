description: Returns the string identifier of an activation function.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.keras.activations.serialize" />
<meta itemprop="path" content="Stable" />
</div>

# tf.keras.activations.serialize

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/activations.py#L513-L578">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Returns the string identifier of an activation function.


<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.keras.activations.serialize(
    activation, use_legacy_format=False
)
</code></pre>



<!-- Placeholder for "Used in" -->


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`activation`<a id="activation"></a>
</td>
<td>
Function object.
</td>
</tr><tr>
<td>
`use_legacy_format`<a id="use_legacy_format"></a>
</td>
<td>
Boolean, whether to use the legacy format for
serialization. Defaults to False.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
String denoting the name attribute of the input function
</td>
</tr>

</table>



#### Example:



```
>>> tf.keras.activations.serialize(tf.keras.activations.tanh)
'tanh'
>>> tf.keras.activations.serialize(tf.keras.activations.sigmoid)
'sigmoid'
>>> tf.keras.activations.serialize('abcd')
Traceback (most recent call last):
...
ValueError: Unknown activation function 'abcd' cannot be serialized.
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
The input function is not a valid one.
</td>
</tr>
</table>

