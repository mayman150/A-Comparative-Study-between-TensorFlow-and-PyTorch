description: Returns the compilation flags for compiling with TensorFlow.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.sysconfig.get_compile_flags" />
<meta itemprop="path" content="Stable" />
</div>

# tf.sysconfig.get_compile_flags

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/platform/sysconfig.py">View source</a>



Returns the compilation flags for compiling with TensorFlow.


<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.sysconfig.get_compile_flags`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.sysconfig.get_compile_flags()
</code></pre>



<!-- Placeholder for "Used in" -->

The returned list of arguments can be passed to the compiler for compiling
against TensorFlow headers. The result is platform dependent.

For example, on a typical Linux system with Python 3.7 the following command
prints `['-I/usr/local/lib/python3.7/dist-packages/tensorflow/include',
'-D_GLIBCXX_USE_CXX11_ABI=1', '-DEIGEN_MAX_ALIGN_BYTES=64']`

```
>>> print(tf.sysconfig.get_compile_flags())
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A list of strings for the compiler flags.
</td>
</tr>

</table>

