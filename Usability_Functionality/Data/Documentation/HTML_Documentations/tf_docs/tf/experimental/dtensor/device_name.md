description: Returns the singleton DTensor device's name.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.experimental.dtensor.device_name" />
<meta itemprop="path" content="Stable" />
</div>

# tf.experimental.dtensor.device_name

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/dtensor/python/api.py">View source</a>



Returns the singleton DTensor device's name.


<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.experimental.dtensor.device_name() -> str
</code></pre>



<!-- Placeholder for "Used in" -->

This function can be used in the following way:

```python
import tensorflow as tf

with tf.device(dtensor.device_name()):
  # ...
```