description: Changes the default value of the provided flag object.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.compat.v1.flags.set_default" />
<meta itemprop="path" content="Stable" />
</div>

# tf.compat.v1.flags.set_default

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>



Changes the default value of the provided flag object.


<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.compat.v1.flags.set_default(
    flag_holder: _flagvalues.FlagHolder[_T], value: _T
) -> None
</code></pre>



<!-- Placeholder for "Used in" -->

The flag's current value is also updated if the flag is currently using
the default value, i.e. not specified in the command line, and not set
by FLAGS.name = value.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`flag_holder`<a id="flag_holder"></a>
</td>
<td>
FlagHolder, the flag to modify.
</td>
</tr><tr>
<td>
`value`<a id="value"></a>
</td>
<td>
The new default value.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Raises</h2></th></tr>

<tr>
<td>
`IllegalFlagValueError`<a id="IllegalFlagValueError"></a>
</td>
<td>
Raised when value is not valid.
</td>
</tr>
</table>

