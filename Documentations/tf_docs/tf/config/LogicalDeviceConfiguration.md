description: Configuration class for a logical devices.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.config.LogicalDeviceConfiguration" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__new__"/>
</div>

# tf.config.LogicalDeviceConfiguration

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/eager/context.py">View source</a>



Configuration class for a logical devices.

<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Main aliases</b>
<p>`tf.config.experimental.VirtualDeviceConfiguration`</p>

<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.config.LogicalDeviceConfiguration`, `tf.compat.v1.config.experimental.VirtualDeviceConfiguration`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.config.LogicalDeviceConfiguration(
    memory_limit=None,
    experimental_priority=None,
    experimental_device_ordinal=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

The class specifies the parameters to configure a <a href="../../tf/config/PhysicalDevice.md"><code>tf.config.PhysicalDevice</code></a>
as it is initialized to a <a href="../../tf/config/LogicalDevice.md"><code>tf.config.LogicalDevice</code></a> during runtime
initialization. Not all fields are valid for all device types.

See <a href="../../tf/config/get_logical_device_configuration.md"><code>tf.config.get_logical_device_configuration</code></a> and
<a href="../../tf/config/set_logical_device_configuration.md"><code>tf.config.set_logical_device_configuration</code></a> for usage examples.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Fields</h2></th></tr>

<tr>
<td>
`memory_limit`<a id="memory_limit"></a>
</td>
<td>
(optional) Maximum memory (in MB) to allocate on the virtual
device. Currently only supported for GPUs.
</td>
</tr><tr>
<td>
`experimental_priority`<a id="experimental_priority"></a>
</td>
<td>
(optional) Priority to assign to a virtual device.
Lower values have higher priorities and 0 is the default.
Within a physical GPU, the GPU scheduler will prioritize ops on virtual
devices with higher priority. Currently only supported for Nvidia GPUs.
</td>
</tr><tr>
<td>
`experimental_device_ordinal`<a id="experimental_device_ordinal"></a>
</td>
<td>
(optional) Ordinal number to order the virtual
device.
  LogicalDevice with lower ordinal number will receive a lower device id.
  Physical device id and location in the list is used to break ties.
  Currently only supported for Nvidia GPUs.
</td>
</tr>
</table>





<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`memory_limit`<a id="memory_limit"></a>
</td>
<td>
A `namedtuple` alias for field number 0
</td>
</tr><tr>
<td>
`experimental_priority`<a id="experimental_priority"></a>
</td>
<td>
A `namedtuple` alias for field number 1
</td>
</tr><tr>
<td>
`experimental_device_ordinal`<a id="experimental_device_ordinal"></a>
</td>
<td>
A `namedtuple` alias for field number 2
</td>
</tr>
</table>



