description: Parses a JSON tokenizer configuration and returns a tokenizer instance.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.keras.preprocessing.text.tokenizer_from_json" />
<meta itemprop="path" content="Stable" />
</div>

# tf.keras.preprocessing.text.tokenizer_from_json

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/preprocessing/text.py#L578-L613">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Parses a JSON tokenizer configuration and returns a tokenizer instance.


<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.keras.preprocessing.text.tokenizer_from_json(
    json_string
)
</code></pre>



<!-- Placeholder for "Used in" -->

Deprecated: <a href="../../../../tf/keras/preprocessing/text/Tokenizer.md"><code>tf.keras.preprocessing.text.Tokenizer</code></a> does not operate on
tensors and is not recommended for new code. Prefer
<a href="../../../../tf/keras/layers/TextVectorization.md"><code>tf.keras.layers.TextVectorization</code></a> which provides equivalent functionality
through a layer which accepts <a href="../../../../tf/Tensor.md"><code>tf.Tensor</code></a> input. See the
[text loading tutorial](https://www.tensorflow.org/tutorials/load_data/text)
for an overview of the layer and text handling in tensorflow.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`json_string`<a id="json_string"></a>
</td>
<td>
JSON string encoding a tokenizer configuration.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A Keras Tokenizer instance
</td>
</tr>

</table>

