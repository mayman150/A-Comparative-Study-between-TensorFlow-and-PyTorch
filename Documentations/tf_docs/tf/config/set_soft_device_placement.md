description: Enable or disable soft device placement.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.config.set_soft_device_placement" />
<meta itemprop="path" content="Stable" />
</div>

# tf.config.set_soft_device_placement

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/framework/config.py">View source</a>



Enable or disable soft device placement.


<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.config.set_soft_device_placement`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.config.set_soft_device_placement(
    enabled
)
</code></pre>



<!-- Placeholder for "Used in" -->

If enabled, ops can be placed on different devices than the device explicitly
assigned by the user. This potentially has a large performance cost due to an
increase in data communication between devices.

Some cases where soft_device_placement would modify device assignment are:
  1. no GPU/TPU implementation for the OP
  2. no GPU devices are known or registered
  3. need to co-locate with reftype input(s) which are from CPU
  4. an OP can not be compiled by XLA.  Common for TPU which always requires
       the XLA compiler.

For TPUs, if this option is true, a feature called automatic outside
compilation is enabled. Automatic outside compilation will move uncompilable
ops within a TPU program to instead run on the host. This can be used when
encountering compilation failures due to unsupported ops.

Note: by default soft device placement is enabled when running in eager mode
(for convenience) and disabled in graph mode (for performance).

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`enabled`<a id="enabled"></a>
</td>
<td>
A boolean indicating whether to enable soft placement.
</td>
</tr>
</table>

