description: Registers a :class:Flag object with a :class:FlagValues object.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.compat.v1.flags.DEFINE_flag" />
<meta itemprop="path" content="Stable" />
</div>

# tf.compat.v1.flags.DEFINE_flag

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>



Registers a :class:`Flag` object with a :class:`FlagValues` object.


<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.compat.v1.flags.DEFINE_flag(
    flag, flag_values=_flagvalues.FLAGS, module_name=None, required=False
)
</code></pre>



<!-- Placeholder for "Used in" -->

By default, the global :const:`FLAGS` ``FlagValue`` object is used.

Typical users will use one of the more specialized DEFINE_xxx
functions, such as :func:`DEFINE_string` or :func:`DEFINE_integer`.  But
developers who need to create :class:`Flag` objects themselves should use
this function to register their flags.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`flag`<a id="flag"></a>
</td>
<td>
:class:`Flag`, a flag that is key to the module.
</td>
</tr><tr>
<td>
`flag_values`<a id="flag_values"></a>
</td>
<td>
:class:`FlagValues`, the ``FlagValues`` instance with which the
flag will be registered. This should almost never need to be overridden.
</td>
</tr><tr>
<td>
`module_name`<a id="module_name"></a>
</td>
<td>
str, the name of the Python module declaring this flag. If not
provided, it will be computed using the stack trace of this call.
</td>
</tr><tr>
<td>
`required`<a id="required"></a>
</td>
<td>
bool, is this a required flag. This must be used as a keyword
argument.
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

