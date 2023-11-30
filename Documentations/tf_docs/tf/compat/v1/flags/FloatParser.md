description: Parser of floating point values.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.compat.v1.flags.FloatParser" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="convert"/>
<meta itemprop="property" content="flag_type"/>
<meta itemprop="property" content="is_outside_bounds"/>
<meta itemprop="property" content="parse"/>
<meta itemprop="property" content="number_article"/>
<meta itemprop="property" content="number_name"/>
<meta itemprop="property" content="syntactic_help"/>
</div>

# tf.compat.v1.flags.FloatParser

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>



Parser of floating point values.

Inherits From: [`ArgumentParser`](../../../../tf/compat/v1/flags/ArgumentParser.md)

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.compat.v1.flags.FloatParser(
    lower_bound: Optional[float] = None, upper_bound: Optional[float] = None
) -> None
</code></pre>



<!-- Placeholder for "Used in" -->

Parsed value may be bounded to a given upper and lower bound.

## Methods

<h3 id="convert"><code>convert</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>convert(
    argument: Union[int, float, str]
) -> float
</code></pre>

Returns the float value of argument.


<h3 id="flag_type"><code>flag_type</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>flag_type() -> Text
</code></pre>

See base class.


<h3 id="is_outside_bounds"><code>is_outside_bounds</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>is_outside_bounds(
    val: _N
) -> bool
</code></pre>

Returns whether the value is outside the bounds or not.


<h3 id="parse"><code>parse</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>parse(
    argument: Text
) -> _N
</code></pre>

See base class.






<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Class Variables</h2></th></tr>

<tr>
<td>
number_article<a id="number_article"></a>
</td>
<td>
`'a'`
</td>
</tr><tr>
<td>
number_name<a id="number_name"></a>
</td>
<td>
`'number'`
</td>
</tr><tr>
<td>
syntactic_help<a id="syntactic_help"></a>
</td>
<td>
`'a number'`
</td>
</tr>
</table>

