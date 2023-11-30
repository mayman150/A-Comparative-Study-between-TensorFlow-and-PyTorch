description: Returns activation function given a string identifier.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.keras.activations.deserialize" />
<meta itemprop="path" content="Stable" />
</div>

# tf.keras.activations.deserialize

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/activations.py#L589-L650">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Returns activation function given a string identifier.


<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.keras.activations.deserialize(
    name, custom_objects=None, use_legacy_format=False
)
</code></pre>



<!-- Placeholder for "Used in" -->


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`name`<a id="name"></a>
</td>
<td>
The name of the activation function.
</td>
</tr><tr>
<td>
`custom_objects`<a id="custom_objects"></a>
</td>
<td>
Optional `{function_name: function_obj}`
dictionary listing user-provided activation functions.
</td>
</tr><tr>
<td>
`use_legacy_format`<a id="use_legacy_format"></a>
</td>
<td>
Boolean, whether to use the legacy format for
deserialization. Defaults to False.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
Corresponding activation function.
</td>
</tr>

</table>



#### Example:



```
>>> tf.keras.activations.deserialize('linear')
 <function linear at 0x1239596a8>
>>> tf.keras.activations.deserialize('sigmoid')
 <function sigmoid at 0x123959510>
>>> tf.keras.activations.deserialize('abcd')
Traceback (most recent call last):
...
ValueError: Unknown activation function 'abcd' cannot be deserialized.
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
`Unknown activation function` if the input string does not
denote any defined Tensorflow activation function.
</td>
</tr>
</table>

