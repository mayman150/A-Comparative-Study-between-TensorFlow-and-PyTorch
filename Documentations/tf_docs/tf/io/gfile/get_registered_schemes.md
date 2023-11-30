description: Returns the currently registered filesystem schemes.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.io.gfile.get_registered_schemes" />
<meta itemprop="path" content="Stable" />
</div>

# tf.io.gfile.get_registered_schemes

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/lib/io/file_io.py">View source</a>



Returns the currently registered filesystem schemes.


<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.io.gfile.get_registered_schemes`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.io.gfile.get_registered_schemes()
</code></pre>



<!-- Placeholder for "Used in" -->

The <a href="../../../tf/io/gfile.md"><code>tf.io.gfile</code></a> APIs, in addition to accepting traditional filesystem paths,
also accept file URIs that begin with a scheme. For example, the local
filesystem path `/tmp/tf` can also be addressed as `file:///tmp/tf`. In this
case, the scheme is `file`, followed by `://` and then the path, according to
[URI syntax](https://datatracker.ietf.org/doc/html/rfc3986#section-3).

This function returns the currently registered schemes that will be recognized
by <a href="../../../tf/io/gfile.md"><code>tf.io.gfile</code></a> APIs. This includes both built-in schemes and those
registered by other TensorFlow filesystem implementations, for example those
provided by [TensorFlow I/O](https://github.com/tensorflow/io).

The empty string is always included, and represents the "scheme" for regular
local filesystem paths.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
List of string schemes, e.g. `['', 'file', 'ram']`, in arbitrary order.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Raises</h2></th></tr>

<tr>
<td>
`errors.OpError`<a id="errors.OpError"></a>
</td>
<td>
If the operation fails.
</td>
</tr>
</table>

