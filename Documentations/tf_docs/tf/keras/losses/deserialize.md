description: Deserializes a serialized loss class/function instance.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.keras.losses.deserialize" />
<meta itemprop="path" content="Stable" />
</div>

# tf.keras.losses.deserialize

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/losses.py#L2896-L2923">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Deserializes a serialized loss class/function instance.


<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.keras.losses.deserialize(
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
Loss configuration.
</td>
</tr><tr>
<td>
`custom_objects`<a id="custom_objects"></a>
</td>
<td>
Optional dictionary mapping names (strings) to custom
objects (classes and functions) to be considered during
deserialization.
</td>
</tr><tr>
<td>
`use_legacy_format`<a id="use_legacy_format"></a>
</td>
<td>
Boolean, whether to use the legacy serialization
format. Defaults to `False`.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A Keras `Loss` instance or a loss function.
</td>
</tr>

</table>

