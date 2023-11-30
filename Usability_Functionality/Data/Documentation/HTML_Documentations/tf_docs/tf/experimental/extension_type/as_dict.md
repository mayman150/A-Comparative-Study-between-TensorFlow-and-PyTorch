description: Extracts the attributes of value and their values to a dict format.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.experimental.extension_type.as_dict" />
<meta itemprop="path" content="Stable" />
</div>

# tf.experimental.extension_type.as_dict

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/framework/extension_type.py">View source</a>



Extracts the attributes of `value` and their values to a dict format.


<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.experimental.extension_type.as_dict`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.experimental.extension_type.as_dict(
    value
)
</code></pre>



<!-- Placeholder for "Used in" -->

Unlike `dataclasses.asdict()`, this function is not recursive and in case of
nested `ExtensionType` objects, only the top level object is converted to a
dict.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`value`<a id="value"></a>
</td>
<td>
An `ExtensionType` object.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A dict that contains the attributes of `value` and their values.
</td>
</tr>

</table>

