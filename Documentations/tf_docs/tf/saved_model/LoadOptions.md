description: Options for loading a SavedModel.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.saved_model.LoadOptions" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
</div>

# tf.saved_model.LoadOptions

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/saved_model/load_options.py">View source</a>



Options for loading a SavedModel.

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.saved_model.LoadOptions(
    allow_partial_checkpoint=False,
    experimental_io_device=None,
    experimental_skip_checkpoint=False,
    experimental_variable_policy=None,
    experimental_load_function_aliases=False
)
</code></pre>



<!-- Placeholder for "Used in" -->

This function may be used in the `options` argument in functions that
load a SavedModel (<a href="../../tf/saved_model/load.md"><code>tf.saved_model.load</code></a>, <a href="../../tf/keras/saving/load_model.md"><code>tf.keras.models.load_model</code></a>).

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`allow_partial_checkpoint`<a id="allow_partial_checkpoint"></a>
</td>
<td>
bool. Defaults to `False`. When enabled, allows
the SavedModel checkpoint to not entirely match the loaded object.
</td>
</tr><tr>
<td>
`experimental_io_device`<a id="experimental_io_device"></a>
</td>
<td>
string. Applies in a distributed setting.
Tensorflow device to use to access the filesystem. If `None` (default)
then for each variable the filesystem is accessed from the CPU:0 device
of the host where that variable is assigned. If specified, the
filesystem is instead accessed from that device for all variables.
This is for example useful if you want to load from a local directory,
such as "/tmp" when running in a distributed setting. In that case
pass a device for the host where the "/tmp" directory is accessible.
</td>
</tr><tr>
<td>
`experimental_skip_checkpoint`<a id="experimental_skip_checkpoint"></a>
</td>
<td>
bool. Defaults to `False`. If set to `True`,
checkpoints will not be restored. Note that this in the majority of
cases will generate an unusable model.
</td>
</tr><tr>
<td>
`experimental_variable_policy`<a id="experimental_variable_policy"></a>
</td>
<td>
string. The policy to apply to variables
when loading. This is either a <a href="../../tf/saved_model/experimental/VariablePolicy.md"><code>saved_model.experimental.VariablePolicy</code></a>
enum instance or one of its value strings (case is not important). See
that enum documentation for details. A value of `None` corresponds to
the default policy.
</td>
</tr><tr>
<td>
`experimental_load_function_aliases`<a id="experimental_load_function_aliases"></a>
</td>
<td>
bool. Defaults to `False`. If set to
`True`, a `function_aliases` attribute will be added to the loaded
SavedModel object.
</td>
</tr>
</table>





<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`allow_partial_checkpoint`<a id="allow_partial_checkpoint"></a>
</td>
<td>

</td>
</tr><tr>
<td>
`experimental_io_device`<a id="experimental_io_device"></a>
</td>
<td>

</td>
</tr><tr>
<td>
`experimental_load_function_aliases`<a id="experimental_load_function_aliases"></a>
</td>
<td>

</td>
</tr><tr>
<td>
`experimental_skip_checkpoint`<a id="experimental_skip_checkpoint"></a>
</td>
<td>

</td>
</tr><tr>
<td>
`experimental_variable_policy`<a id="experimental_variable_policy"></a>
</td>
<td>

</td>
</tr>
</table>



