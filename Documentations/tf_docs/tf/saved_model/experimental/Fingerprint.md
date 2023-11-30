description: The SavedModel fingerprint.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.saved_model.experimental.Fingerprint" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__eq__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="from_proto"/>
<meta itemprop="property" content="singleprint"/>
</div>

# tf.saved_model.experimental.Fingerprint

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/saved_model/fingerprinting.py">View source</a>



The SavedModel fingerprint.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.saved_model.experimental.Fingerprint(
    saved_model_checksum=None,
    graph_def_program_hash=None,
    signature_def_hash=None,
    saved_object_graph_hash=None,
    checkpoint_hash=None,
    version=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

Each attribute of this class is named after a field name in the
FingerprintDef proto and contains the value of the respective field in the
protobuf.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`saved_model_checksum`<a id="saved_model_checksum"></a>
</td>
<td>
Value of the`saved_model_checksum`.
</td>
</tr><tr>
<td>
`graph_def_program_hash`<a id="graph_def_program_hash"></a>
</td>
<td>
Value of the `graph_def_program_hash`.
</td>
</tr><tr>
<td>
`signature_def_hash`<a id="signature_def_hash"></a>
</td>
<td>
Value of the `signature_def_hash`.
</td>
</tr><tr>
<td>
`saved_object_graph_hash`<a id="saved_object_graph_hash"></a>
</td>
<td>
Value of the `saved_object_graph_hash`.
</td>
</tr><tr>
<td>
`checkpoint_hash`<a id="checkpoint_hash"></a>
</td>
<td>
Value of the `checkpoint_hash`.
</td>
</tr><tr>
<td>
`version`<a id="version"></a>
</td>
<td>
Value of the producer field of the VersionDef.
</td>
</tr>
</table>





<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`saved_model_checksum`<a id="saved_model_checksum"></a>
</td>
<td>
A uint64 containing the `saved_model_checksum`.
</td>
</tr><tr>
<td>
`graph_def_program_hash`<a id="graph_def_program_hash"></a>
</td>
<td>
A uint64 containing `graph_def_program_hash`.
</td>
</tr><tr>
<td>
`signature_def_hash`<a id="signature_def_hash"></a>
</td>
<td>
A uint64 containing the `signature_def_hash`.
</td>
</tr><tr>
<td>
`saved_object_graph_hash`<a id="saved_object_graph_hash"></a>
</td>
<td>
A uint64 containing the `saved_object_graph_hash`.
</td>
</tr><tr>
<td>
`checkpoint_hash`<a id="checkpoint_hash"></a>
</td>
<td>
A uint64 containing the`checkpoint_hash`.
</td>
</tr><tr>
<td>
`version`<a id="version"></a>
</td>
<td>
An int32 containing the producer field of the VersionDef.
</td>
</tr>
</table>



## Methods

<h3 id="from_proto"><code>from_proto</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/saved_model/fingerprinting.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@classmethod</code>
<code>from_proto(
    proto
)
</code></pre>

Constructs Fingerprint object from protocol buffer message.


<h3 id="singleprint"><code>singleprint</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/saved_model/fingerprinting.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>singleprint()
</code></pre>

Canonical fingerprinting ID for a SavedModel.

Uniquely identifies a SavedModel based on the regularized fingerprint
attributes. (saved_model_checksum is sensitive to immaterial changes and
thus non-deterministic.)

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
The string concatenation of `graph_def_program_hash`,
`signature_def_hash`, `saved_object_graph_hash`, and `checkpoint_hash`
fingerprint attributes (separated by '/').
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>

<tr>
<td>
`ValueError`
</td>
<td>
If the fingerprint fields cannot be used to construct the
singleprint.
</td>
</tr>
</table>



<h3 id="__eq__"><code>__eq__</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/saved_model/fingerprinting.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__eq__(
    other
)
</code></pre>

Return self==value.




