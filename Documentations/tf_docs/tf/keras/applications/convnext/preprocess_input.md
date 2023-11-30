description: A placeholder method for backward compatibility.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.keras.applications.convnext.preprocess_input" />
<meta itemprop="path" content="Stable" />
</div>

# tf.keras.applications.convnext.preprocess_input

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/applications/convnext.py#L747-L766">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



A placeholder method for backward compatibility.


<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.keras.applications.convnext.preprocess_input(
    x, data_format=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

The preprocessing logic has been included in the convnext model
implementation. Users are no longer required to call this method to
normalize the input data. This method does nothing and only kept as a
placeholder to align the API surface between old and new version of model.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`x`<a id="x"></a>
</td>
<td>
A floating point `numpy.array` or a <a href="../../../../tf/Tensor.md"><code>tf.Tensor</code></a>.
</td>
</tr><tr>
<td>
`data_format`<a id="data_format"></a>
</td>
<td>
Optional data format of the image tensor/array. `None` means
the global setting <a href="../../../../tf/keras/backend/image_data_format.md"><code>tf.keras.backend.image_data_format()</code></a> is used
(unless you changed it, it uses "channels_last").
Defaults to `None`.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
Unchanged `numpy.array` or <a href="../../../../tf/Tensor.md"><code>tf.Tensor</code></a>.
</td>
</tr>

</table>

