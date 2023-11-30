description: Ensures that only one flag among flag_names is True.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.compat.v1.flags.mark_bool_flags_as_mutual_exclusive" />
<meta itemprop="path" content="Stable" />
</div>

# tf.compat.v1.flags.mark_bool_flags_as_mutual_exclusive

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>



Ensures that only one flag among flag_names is True.


<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.compat.v1.flags.mark_bool_flags_as_mutual_exclusive(
    flag_names, required=False, flag_values=_flagvalues.FLAGS
)
</code></pre>



<!-- Placeholder for "Used in" -->


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`flag_names`<a id="flag_names"></a>
</td>
<td>
[str | FlagHolder], names or holders of flags.
Positional-only parameter.
</td>
</tr><tr>
<td>
`required`<a id="required"></a>
</td>
<td>
bool. If true, exactly one flag must be True. Otherwise, at most
one flag can be True, and it is valid for all flags to be False.
</td>
</tr><tr>
<td>
`flag_values`<a id="flag_values"></a>
</td>
<td>
flags.FlagValues, optional FlagValues instance where the flags
are defined.
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
Raised when multiple FlagValues are used in the same
invocation. This can occur when FlagHolders have different `_flagvalues`
or when str-type flag_names entries are present and the `flag_values`
argument does not match that of provided FlagHolder(s).
</td>
</tr>
</table>

