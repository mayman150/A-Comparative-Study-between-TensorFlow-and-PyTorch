description: Public API for tf._api.v2.compat namespace

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.compat" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="bytes_or_text_types"/>
<meta itemprop="property" content="complex_types"/>
<meta itemprop="property" content="integral_types"/>
<meta itemprop="property" content="real_types"/>
</div>

# Module: tf.compat

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>



Public API for tf._api.v2.compat namespace



## Modules

[`v1`](../tf/compat/v1.md) module: Bring in all of the public TensorFlow interface into this module.

## Functions

[`as_bytes(...)`](../tf/compat/as_bytes.md): Converts `bytearray`, `bytes`, or unicode python input types to `bytes`.

[`as_str(...)`](../tf/compat/as_str.md)

[`as_str_any(...)`](../tf/compat/as_str_any.md): Converts input to `str` type.

[`as_text(...)`](../tf/compat/as_text.md): Converts any string-like python input types to unicode.

[`dimension_at_index(...)`](../tf/compat/dimension_at_index.md): Compatibility utility required to allow for both V1 and V2 behavior in TF.

[`dimension_value(...)`](../tf/compat/dimension_value.md): Compatibility utility required to allow for both V1 and V2 behavior in TF.

[`forward_compatibility_horizon(...)`](../tf/compat/forward_compatibility_horizon.md): Context manager for testing forward compatibility of generated graphs.

[`forward_compatible(...)`](../tf/compat/forward_compatible.md): Return true if the forward compatibility window has expired.

[`path_to_str(...)`](../tf/compat/path_to_str.md): Converts input which is a `PathLike` object to `str` type.



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Other Members</h2></th></tr>

<tr>
<td>
bytes_or_text_types<a id="bytes_or_text_types"></a>
</td>
<td>
`(<class 'bytes'>, <class 'str'>)`
</td>
</tr><tr>
<td>
complex_types<a id="complex_types"></a>
</td>
<td>
`(<class 'numbers.Complex'>, <class 'numpy.number'>)`
</td>
</tr><tr>
<td>
integral_types<a id="integral_types"></a>
</td>
<td>
`(<class 'numbers.Integral'>, <class 'numpy.integer'>)`
</td>
</tr><tr>
<td>
real_types<a id="real_types"></a>
</td>
<td>
`(<class 'numbers.Real'>, <class 'numpy.integer'>, <class 'numpy.floating'>)`
</td>
</tr>
</table>

