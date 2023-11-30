description: Enables experimental MLIR-Based TensorFlow Compiler Bridge.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.config.experimental.enable_mlir_bridge" />
<meta itemprop="path" content="Stable" />
</div>

# tf.config.experimental.enable_mlir_bridge

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/framework/config.py">View source</a>



Enables experimental MLIR-Based TensorFlow Compiler Bridge.


<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.config.experimental.enable_mlir_bridge`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.config.experimental.enable_mlir_bridge()
</code></pre>



<!-- Placeholder for "Used in" -->

TensorFlow Compiler Bridge (TF Bridge) is responsible for translating parts
of TensorFlow graph into a form that can be accepted as an input by a backend
compiler such as XLA.