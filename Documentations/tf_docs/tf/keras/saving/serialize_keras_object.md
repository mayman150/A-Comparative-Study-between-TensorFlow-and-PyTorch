description: Retrieve the config dict by serializing the Keras object.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.keras.saving.serialize_keras_object" />
<meta itemprop="path" content="Stable" />
</div>

# tf.keras.saving.serialize_keras_object

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/saving/serialization_lib.py#L128-L289">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Retrieve the config dict by serializing the Keras object.


<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`tf.keras.utils.serialize_keras_object`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.keras.saving.serialize_keras_object(
    obj
)
</code></pre>



<!-- Placeholder for "Used in" -->

`serialize_keras_object()` serializes a Keras object to a python dictionary
that represents the object, and is a reciprocal function of
`deserialize_keras_object()`. See `deserialize_keras_object()` for more
information about the config format.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`obj`<a id="obj"></a>
</td>
<td>
the Keras object to serialize.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A python dict that represents the object. The python dict can be
deserialized via `deserialize_keras_object()`.
</td>
</tr>

</table>

