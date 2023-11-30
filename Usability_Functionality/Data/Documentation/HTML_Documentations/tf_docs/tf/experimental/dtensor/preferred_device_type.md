description: Returns the preferred device type for the accelerators.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.experimental.dtensor.preferred_device_type" />
<meta itemprop="path" content="Stable" />
</div>

# tf.experimental.dtensor.preferred_device_type

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/dtensor/python/config.py">View source</a>



Returns the preferred device type for the accelerators.


<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.experimental.dtensor.preferred_device_type() -> str
</code></pre>



<!-- Placeholder for "Used in" -->

The returned device type is determined by checking the first present device
type from all supported device types in the order of 'TPU', 'GPU', 'CPU'.