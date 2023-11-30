description: Parses a yaml model configuration file and returns a model instance.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.keras.models.model_from_yaml" />
<meta itemprop="path" content="Stable" />
</div>

# tf.keras.models.model_from_yaml

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/saving/legacy/model_config.py#L74-L97">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Parses a yaml model configuration file and returns a model instance.


<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.keras.models.model_from_yaml(
    yaml_string, custom_objects=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

Note: Since TF 2.6, this method is no longer supported and will raise a
RuntimeError.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`yaml_string`<a id="yaml_string"></a>
</td>
<td>
YAML string or open file encoding a model configuration.
</td>
</tr><tr>
<td>
`custom_objects`<a id="custom_objects"></a>
</td>
<td>
Optional dictionary mapping names
(strings) to custom classes or functions to be
considered during deserialization.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A Keras model instance (uncompiled).
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Raises</h2></th></tr>

<tr>
<td>
`RuntimeError`<a id="RuntimeError"></a>
</td>
<td>
announces that the method poses a security risk
</td>
</tr>
</table>

