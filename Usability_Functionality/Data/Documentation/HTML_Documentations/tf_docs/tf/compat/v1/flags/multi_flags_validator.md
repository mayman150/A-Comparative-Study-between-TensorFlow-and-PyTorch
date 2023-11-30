description: A function decorator for defining a multi-flag validator.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.compat.v1.flags.multi_flags_validator" />
<meta itemprop="path" content="Stable" />
</div>

# tf.compat.v1.flags.multi_flags_validator

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>



A function decorator for defining a multi-flag validator.


<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.compat.v1.flags.multi_flags_validator(
    flag_names,
    message=&#x27;Flag validation failed&#x27;,
    flag_values=_flagvalues.FLAGS
)
</code></pre>



<!-- Placeholder for "Used in" -->

Registers the decorated function as a validator for flag_names, e.g.::

    @flags.multi_flags_validator(['foo', 'bar'])
    def _CheckFooBar(flags_dict):
      ...

See :func:`register_multi_flags_validator` for the specification of checker
function.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`flag_names`<a id="flag_names"></a>
</td>
<td>
[str | FlagHolder], a list of the flag names or holders to be
checked. Positional-only parameter.
</td>
</tr><tr>
<td>
`message`<a id="message"></a>
</td>
<td>
str, error text to be shown to the user if checker returns False.
If checker raises flags.ValidationError, message from the raised
error will be shown.
</td>
</tr><tr>
<td>
`flag_values`<a id="flag_values"></a>
</td>
<td>
flags.FlagValues, optional FlagValues instance to validate
against.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A function decorator that registers its function argument as a validator.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Raises</h2></th></tr>

<tr>
<td>
`AttributeError`<a id="AttributeError"></a>
</td>
<td>
Raised when a flag is not registered as a valid flag name.
</td>
</tr>
</table>

