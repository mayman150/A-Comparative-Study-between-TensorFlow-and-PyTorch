description: Declares one flag as key to the current module.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.compat.v1.flags.declare_key_flag" />
<meta itemprop="path" content="Stable" />
</div>

# tf.compat.v1.flags.declare_key_flag

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>



Declares one flag as key to the current module.


<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.compat.v1.flags.declare_key_flag(
    flag_name: Union[Text, <a href="../../../../tf/compat/v1/flags/FlagHolder.md"><code>tf.compat.v1.flags.FlagHolder</code></a>],
    flag_values: <a href="../../../../tf/compat/v1/flags/FlagValues.md"><code>tf.compat.v1.flags.FlagValues</code></a> = _flagvalues.FLAGS
) -> None
</code></pre>



<!-- Placeholder for "Used in" -->

Key flags are flags that are deemed really important for a module.
They are important when listing help messages; e.g., if the
--helpshort command-line flag is used, then only the key flags of the
main module are listed (instead of all flags, as in the case of
--helpfull).

Sample usage::

    flags.declare_key_flag('flag_1')

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`flag_name`<a id="flag_name"></a>
</td>
<td>
str | :class:`FlagHolder`, the name or holder of an already
declared flag. (Redeclaring flags as key, including flags implicitly key
because they were declared in this module, is a no-op.)
Positional-only parameter.
</td>
</tr><tr>
<td>
`flag_values`<a id="flag_values"></a>
</td>
<td>
:class:`FlagValues`, the FlagValues instance in which the
flag will be declared as a key flag. This should almost never need to be
overridden.
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
Raised if flag_name not defined as a Python flag.
</td>
</tr>
</table>

