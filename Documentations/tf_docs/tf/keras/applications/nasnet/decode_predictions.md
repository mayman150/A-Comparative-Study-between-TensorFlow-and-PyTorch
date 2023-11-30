description: Decodes the prediction of an ImageNet model.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.keras.applications.nasnet.decode_predictions" />
<meta itemprop="path" content="Stable" />
</div>

# tf.keras.applications.nasnet.decode_predictions

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/applications/nasnet.py#L900-L902">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Decodes the prediction of an ImageNet model.


<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.keras.applications.nasnet.decode_predictions(
    preds, top=5
)
</code></pre>



<!-- Placeholder for "Used in" -->


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`preds`<a id="preds"></a>
</td>
<td>
Numpy array encoding a batch of predictions.
</td>
</tr><tr>
<td>
`top`<a id="top"></a>
</td>
<td>
Integer, how many top-guesses to return. Defaults to 5.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A list of lists of top class prediction tuples
`(class_name, class_description, score)`.
One list of tuples per sample in batch input.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Raises</h2></th></tr>

<tr>
<td>
`ValueError`<a id="ValueError"></a>
</td>
<td>
In case of invalid shape of the `pred` array
(must be 2D).
</td>
</tr>
</table>

