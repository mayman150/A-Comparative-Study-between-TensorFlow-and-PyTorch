description: Reads the fingerprint of a SavedModel in export_dir.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.saved_model.experimental.read_fingerprint" />
<meta itemprop="path" content="Stable" />
</div>

# tf.saved_model.experimental.read_fingerprint

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/saved_model/fingerprinting.py">View source</a>



Reads the fingerprint of a SavedModel in `export_dir`.


<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.saved_model.experimental.read_fingerprint(
    export_dir
)
</code></pre>



<!-- Placeholder for "Used in" -->

Returns a <a href="../../../tf/saved_model/experimental/Fingerprint.md"><code>tf.saved_model.experimental.Fingerprint</code></a> object that contains
the values of the SavedModel fingerprint, which is persisted on disk in the
`fingerprint.pb` file in the `export_dir`.

Read more about fingerprints in the SavedModel guide at
https://www.tensorflow.org/guide/saved_model.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`export_dir`<a id="export_dir"></a>
</td>
<td>
The directory that contains the SavedModel.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A <a href="../../../tf/saved_model/experimental/Fingerprint.md"><code>tf.saved_model.experimental.Fingerprint</code></a>.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Raises</h2></th></tr>

<tr>
<td>
`FileNotFoundError`<a id="FileNotFoundError"></a>
</td>
<td>
If no or an invalid fingerprint is found.
</td>
</tr>
</table>

