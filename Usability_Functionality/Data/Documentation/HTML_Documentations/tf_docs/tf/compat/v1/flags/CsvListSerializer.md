description: Base class for generating string representations of a flag value.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.compat.v1.flags.CsvListSerializer" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="serialize"/>
</div>

# tf.compat.v1.flags.CsvListSerializer

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>



Base class for generating string representations of a flag value.

Inherits From: [`ListSerializer`](../../../../tf/compat/v1/flags/ListSerializer.md), [`ArgumentSerializer`](../../../../tf/compat/v1/flags/ArgumentSerializer.md)

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.compat.v1.flags.CsvListSerializer(
    list_sep: Text
) -> None
</code></pre>



<!-- Placeholder for "Used in" -->


## Methods

<h3 id="serialize"><code>serialize</code></h3>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>serialize(
    value: List[Text]
) -> Text
</code></pre>

Serializes a list as a CSV string or unicode.




