description: Registers a flag whose value is a whitespace-separated list of strings.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.compat.v1.flags.DEFINE_spaceseplist" />
<meta itemprop="path" content="Stable" />
</div>

# tf.compat.v1.flags.DEFINE_spaceseplist

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>



Registers a flag whose value is a whitespace-separated list of strings.


<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.compat.v1.flags.DEFINE_spaceseplist(
    name,
    default,
    help,
    comma_compat=False,
    flag_values=_flagvalues.FLAGS,
    required=False,
    **args
)
</code></pre>



<!-- Placeholder for "Used in" -->

Any whitespace can be used as a separator.

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
list|str|None, the default value of the flag.
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
`comma_compat`<a id="comma_compat"></a>
</td>
<td>
bool - Whether to support comma as an additional separator. If
false then only whitespace is supported.  This is intended only for
backwards compatibility with flags that used to be comma-separated.
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
Dictionary with extra keyword args that are passed to the
``Flag.__init__``.
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

