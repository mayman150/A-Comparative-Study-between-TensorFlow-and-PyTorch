description: Ensures that only one flag among flag_names is not None.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.compat.v1.flags.mark_flags_as_mutual_exclusive" />
<meta itemprop="path" content="Stable" />
</div>

# tf.compat.v1.flags.mark_flags_as_mutual_exclusive

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>



Ensures that only one flag among flag_names is not None.


<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.compat.v1.flags.mark_flags_as_mutual_exclusive(
    flag_names, required=False, flag_values=_flagvalues.FLAGS
)
</code></pre>



<!-- Placeholder for "Used in" -->

Important note: This validator checks if flag values are ``None``, and it does
not distinguish between default and explicit values. Therefore, this validator
does not make sense when applied to flags with default values other than None,
including other false values (e.g. ``False``, ``0``, ``''``, ``[]``). That
includes multi flags with a default value of ``[]`` instead of None.

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
bool. If true, exactly one of the flags must have a value other
than None. Otherwise, at most one of the flags can have a value other
than None, and it is valid for all of the flags to be None.
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

