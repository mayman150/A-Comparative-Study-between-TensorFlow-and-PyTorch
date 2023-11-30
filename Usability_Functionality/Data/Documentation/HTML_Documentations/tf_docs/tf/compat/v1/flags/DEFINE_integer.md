description: Registers a flag whose value must be an integer.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.compat.v1.flags.DEFINE_integer" />
<meta itemprop="path" content="Stable" />
</div>

# tf.compat.v1.flags.DEFINE_integer

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>



Registers a flag whose value must be an integer.


<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.compat.v1.flags.DEFINE_integer(
    name,
    default,
    help,
    lower_bound=None,
    upper_bound=None,
    flag_values=_flagvalues.FLAGS,
    required=False,
    **args
)
</code></pre>



<!-- Placeholder for "Used in" -->

If ``lower_bound``, or ``upper_bound`` are set, then this flag must be
within the given range.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`name`<a id="name"></a>
</td>
<td>
str, the flag name.
</td>
</tr><tr>
<td>
`default`<a id="default"></a>
</td>
<td>
int|str|None, the default value of the flag.
</td>
</tr><tr>
<td>
`help`<a id="help"></a>
</td>
<td>
str, the help message.
</td>
</tr><tr>
<td>
`lower_bound`<a id="lower_bound"></a>
</td>
<td>
int, min value of the flag.
</td>
</tr><tr>
<td>
`upper_bound`<a id="upper_bound"></a>
</td>
<td>
int, max value of the flag.
</td>
</tr><tr>
<td>
`flag_values`<a id="flag_values"></a>
</td>
<td>
:class:`FlagValues`, the FlagValues instance with which the
flag will be registered. This should almost never need to be overridden.
</td>
</tr><tr>
<td>
`required`<a id="required"></a>
</td>
<td>
bool, is this a required flag. This must be used as a keyword
argument.
</td>
</tr><tr>
<td>
`**args`<a id="**args"></a>
</td>
<td>
dict, the extra keyword args that are passed to :func:`DEFINE`.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
a handle to defined flag.
</td>
</tr>

</table>

