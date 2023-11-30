description: Serializes a Layer object into a JSON-compatible representation.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.keras.layers.serialize" />
<meta itemprop="path" content="Stable" />
</div>

# tf.keras.layers.serialize

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/layers/serialization.py#L191-L222">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Serializes a `Layer` object into a JSON-compatible representation.


<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.keras.layers.serialize(
    layer, use_legacy_format=False
)
</code></pre>



<!-- Placeholder for "Used in" -->


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`layer`<a id="layer"></a>
</td>
<td>
The `Layer` object to serialize.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A JSON-serializable dict representing the object's config.
</td>
</tr>

</table>



#### Example:



```python
from pprint import pprint
model = tf.keras.models.Sequential()
model.add(tf.keras.Input(shape=(16,)))
model.add(tf.keras.layers.Dense(32, activation='relu'))

pprint(tf.keras.layers.serialize(model))
# prints the configuration of the model, as a dict.