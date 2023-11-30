description: Convert a dict of values into process call parameters.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.compat.v1.flags.flag_dict_to_args" />
<meta itemprop="path" content="Stable" />
</div>

# tf.compat.v1.flags.flag_dict_to_args

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>



Convert a dict of values into process call parameters.


<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.compat.v1.flags.flag_dict_to_args(
    flag_map: Dict[str, Any], multi_flags: Optional[Set[str]] = None
) -> Iterable[str]
</code></pre>



<!-- Placeholder for "Used in" -->

This method is used to convert a dictionary into a sequence of parameters
for a binary that parses arguments using this module.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`flag_map`<a id="flag_map"></a>
</td>
<td>
dict, a mapping where the keys are flag names (strings).
values are treated according to their type:

* If value is ``None``, then only the name is emitted.
* If value is ``True``, then only the name is emitted.
* If value is ``False``, then only the name prepended with 'no' is
  emitted.
* If value is a string then ``--name=value`` is emitted.
* If value is a collection, this will emit
  ``--name=value1,value2,value3``, unless the flag name is in
  ``multi_flags``, in which case this will emit
  ``--name=value1 --name=value2 --name=value3``.
* Everything else is converted to string an passed as such.
</td>
</tr><tr>
<td>
`multi_flags`<a id="multi_flags"></a>
</td>
<td>
set, names (strings) of flags that should be treated as
multi-flags.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Yields</h2></th></tr>
<tr class="alt">
<td colspan="2">
sequence of string suitable for a subprocess execution.
</td>
</tr>

</table>

