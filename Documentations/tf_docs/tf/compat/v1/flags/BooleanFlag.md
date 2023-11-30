description: Basic boolean flag.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.compat.v1.flags.BooleanFlag" />
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

# tf.compat.v1.flags.BooleanFlag

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>



Basic boolean flag.

Inherits From: [`Flag`](../../../../tf/compat/v1/flags/Flag.md)

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.compat.v1.flags.BooleanFlag(
    name: Text,
    default: Union[Optional[bool], Text],
    help: Optional[Text],
    short_name: Optional[Text] = None,
    **args
) -> None
</code></pre>



<!-- Placeholder for "Used in" -->

Boolean flags do not take any arguments, and their value is either
``True`` (1) or ``False`` (0).  The false value is specified on the command
line by prepending the word ``'no'`` to either the long or the short flag
name.

For example, if a Boolean flag was created whose long name was
``'update'`` and whose short name was ``'x'``, then this flag could be
explicitly unset through either ``--noupdate`` or ``--nox``.



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`value`<a id="value"></a>
</td>
<td>

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




