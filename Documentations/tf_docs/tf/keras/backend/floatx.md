description: Returns the default float type, as a string.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.keras.backend.floatx" />
<meta itemprop="path" content="Stable" />
</div>

# tf.keras.backend.floatx

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/backend_config.py#L66-L79">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Returns the default float type, as a string.


<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.keras.backend.floatx()
</code></pre>



<!-- Placeholder for "Used in" -->

E.g. `'float16'`, `'float32'`, `'float64'`.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
String, the current default float type.
</td>
</tr>

</table>



#### Example:


```
>>> tf.keras.backend.floatx()
'float32'
```