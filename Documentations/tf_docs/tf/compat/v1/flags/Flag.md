description: Information about a command-line flag.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.compat.v1.flags.Flag" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__bool__"/>
<meta itemprop="property" content="__eq__"/>
<meta itemprop="property" content="__ge__"/>
<meta itemprop="property" content="__gt__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="__le__"/>
<meta itemprop="property" content="__lt__"/>
<meta itemprop="property" content="flag_type"/>
<meta itemprop="property" content="parse"/>
<meta itemprop="property" content="serialize"/>
<meta itemprop="property" content="unparse"/>
</div>

# tf.compat.v1.flags.Flag

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>



Information about a command-line flag.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.compat.v1.flags.Flag(
    parser: _argument_parser.ArgumentParser[_T],
    serializer: Optional[_argument_parser.ArgumentSerializer[_T]],
    name: Text,
    default: Union[Optional[_T], Text],
    help_string: Optional[Text],
    short_name: Optional[Text] = None,
    boolean: bool = False,
    allow_override: bool = False,
    allow_override_cpp: bool = False,
    allow_hide_cpp: bool = False,
    allow_overwrite: bool = True,
    allow_using_method_names: bool = False
) -> None
</code></pre>



<!-- Placeholder for "Used in" -->

The only public method of a ``Flag`` object is :meth:`parse`, but it is
typically only called by a :class:`~absl.flags.FlagValues` object.  The
:meth:`parse` method is a thin wrapper around the
:meth:`ArgumentParser.parse()<absl.flags.ArgumentParser.parse>` method.  The
parsed value is saved in ``.value``, and the ``.present`` attribute is
updated.  If this flag was already present, an Error is raised.

:meth:`parse` is also called during ``__init__`` to parse the default value
and initialize the ``.value`` attribute.  This enables other python modules to
safely use flags even if the ``__main__`` module neglects to parse the
command line arguments.  The ``.present`` attribute is cleared after
``__init__`` parsing.  If the default value is set to ``None``, then the
``__init__`` parsing step is skipped and the ``.value`` attribute is
initialized to None.

Note: The default value is also presented to the user in the help
string, so it is important that it be a legal value for this flag.



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`name`<a id="name"></a>
</td>
<td>
the name for this flag
</td>
</tr><tr>
<td>
`default`<a id="default"></a>
</td>
<td>
the default value for this flag
</td>
</tr><tr>
<td>
`default_unparsed`<a id="default_unparsed"></a>
</td>
<td>
the unparsed default value for this flag.
</td>
</tr><tr>
<td>
`default_as_str`<a id="default_as_str"></a>
</td>
<td>
default value as repr'd string, e.g., "'true'"
(or None)
</td>
</tr><tr>
<td>
`value`<a id="value"></a>
</td>
<td>
the most recent parsed value of this flag set by :meth:`parse`
</td>
</tr><tr>
<td>
`help`<a id="help"></a>
</td>
<td>
a help string or None if no help is available
</td>
</tr><tr>
<td>
`short_name`<a id="short_name"></a>
</td>
<td>
the single letter alias for this flag (or None)
</td>
</tr><tr>
<td>
`boolean`<a id="boolean"></a>
</td>
<td>
if 'true', this flag does not accept arguments
</td>
</tr><tr>
<td>
`present`<a id="present"></a>
</td>
<td>
true if this flag was parsed from command line flags
</td>
</tr><tr>
<td>
`parser`<a id="parser"></a>
</td>
<td>
an :class:`~absl.flags.ArgumentParser` object
</td>
</tr><tr>
<td>
`serializer`<a id="serializer"></a>
</td>
<td>
an ArgumentSerializer object
</td>
</tr><tr>
<td>
`allow_override`<a id="allow_override"></a>
</td>
<td>
the flag may be redefined without raising an error,
and newly defined flag overrides the old one.
</td>
</tr><tr>
<td>
`allow_override_cpp`<a id="allow_override_cpp"></a>
</td>
<td>
use the flag from C++ if available the flag
definition is replaced by the C++ flag after init
</td>
</tr><tr>
<td>
`allow_hide_cpp`<a id="allow_hide_cpp"></a>
</td>
<td>
use the Python flag despite having a C++ flag with
the same name (ignore the C++ flag)
</td>
</tr><tr>
<td>
`using_default_value`<a id="using_default_value"></a>
</td>
<td>
the flag value has not been set by user
</td>
</tr><tr>
<td>
`allow_overwrite`<a id="allow_overwrite"></a>
</td>
<td>
the flag may be parsed more than once without
raising an error, the last set value will be used
</td>
</tr><tr>
<td>
`allow_using_method_names`<a id="allow_using_method_names"></a>
</td>
<td>
whether this flag can be defined even if
it has a name that conflicts with a FlagValues method.
</td>
</tr><tr>
<td>
`validators`<a id="validators"></a>
</td>
<td>
list of the flag validators.
</td>
</tr>
</table>



## Methods

<h3 id="flag_type"><code>flag_type</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>flag_type() -> Text
</code></pre>

Returns a str that describes the type of the flag.

NOTE: we use strings, and not the types.*Type constants because
our flags can have more exotic types, e.g., 'comma separated list
of strings', 'whitespace separated list of strings', etc.

<h3 id="parse"><code>parse</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>parse(
    argument: Union[Text, Optional[_T]]
) -> None
</code></pre>

Parses string and sets flag value.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`argument`
</td>
<td>
str or the correct flag value type, argument to be parsed.
</td>
</tr>
</table>



<h3 id="serialize"><code>serialize</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>serialize() -> Text
</code></pre>

Serializes the flag.


<h3 id="unparse"><code>unparse</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>unparse() -> None
</code></pre>




<h3 id="__bool__"><code>__bool__</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__bool__()
</code></pre>




<h3 id="__eq__"><code>__eq__</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__eq__(
    other
)
</code></pre>

Return self==value.


<h3 id="__ge__"><code>__ge__</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__ge__(
    other, NotImplemented=NotImplemented
)
</code></pre>

Return a >= b.  Computed by @total_ordering from (not a < b).


<h3 id="__gt__"><code>__gt__</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__gt__(
    other, NotImplemented=NotImplemented
)
</code></pre>

Return a > b.  Computed by @total_ordering from (not a < b) and (a != b).


<h3 id="__le__"><code>__le__</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__le__(
    other, NotImplemented=NotImplemented
)
</code></pre>

Return a <= b.  Computed by @total_ordering from (a < b) or (a == b).


<h3 id="__lt__"><code>__lt__</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__lt__(
    other
)
</code></pre>

Return self<value.




