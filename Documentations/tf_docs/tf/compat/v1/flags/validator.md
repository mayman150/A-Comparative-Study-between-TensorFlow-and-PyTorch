description: A function decorator for defining a flag validator.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.compat.v1.flags.validator" />
<meta itemprop="path" content="Stable" />
</div>

# tf.compat.v1.flags.validator

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>



A function decorator for defining a flag validator.


<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.compat.v1.flags.validator(
    flag_name,
    message=&#x27;Flag validation failed&#x27;,
    flag_values=_flagvalues.FLAGS
)
</code></pre>



<!-- Placeholder for "Used in" -->

Registers the decorated function as a validator for flag_name, e.g.::

    @flags.validator('foo')
    def _CheckFoo(foo):
      ...

See :func:`register_validator` for the specification of checker function.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`flag_name`<a id="flag_name"></a>
</td>
<td>
str | FlagHolder, name or holder of the flag to be checked.
Positional-only parameter.
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
Raised when flag_name is not registered as a valid flag
name.
</td>
</tr>
</table>

