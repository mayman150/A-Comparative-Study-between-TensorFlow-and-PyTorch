description: Configuration for parsing a fixed-length input feature.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.io.FixedLenFeature" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__new__"/>
</div>

# tf.io.FixedLenFeature

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/ops/parsing_config.py">View source</a>



Configuration for parsing a fixed-length input feature.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.FixedLenFeature`, `tf.compat.v1.io.FixedLenFeature`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.io.FixedLenFeature(
    shape, dtype, default_value=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

To treat sparse input as dense, provide a `default_value`; otherwise,
the parse functions will fail on any examples missing this feature.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Fields</h2></th></tr>

<tr>
<td>
`shape`<a id="shape"></a>
</td>
<td>
Shape of input data.
</td>
</tr><tr>
<td>
`dtype`<a id="dtype"></a>
</td>
<td>
Data type of input.
</td>
</tr><tr>
<td>
`default_value`<a id="default_value"></a>
</td>
<td>
Value to be used if an example is missing this feature. It
must be compatible with `dtype` and of the specified `shape`.
</td>
</tr>
</table>





<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`shape`<a id="shape"></a>
</td>
<td>
A `namedtuple` alias for field number 0
</td>
</tr><tr>
<td>
`dtype`<a id="dtype"></a>
</td>
<td>
A `namedtuple` alias for field number 1
</td>
</tr><tr>
<td>
`default_value`<a id="default_value"></a>
</td>
<td>
A `namedtuple` alias for field number 2
</td>
</tr>
</table>



