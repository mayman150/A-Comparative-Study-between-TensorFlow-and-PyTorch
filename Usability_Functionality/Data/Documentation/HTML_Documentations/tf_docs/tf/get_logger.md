description: Return TF logger instance.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.get_logger" />
<meta itemprop="path" content="Stable" />
</div>

# tf.get_logger

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/platform/tf_logging.py">View source</a>



Return TF logger instance.


<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.get_logger`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.get_logger()
</code></pre>



<!-- Placeholder for "Used in" -->


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
An instance of the Python logging library Logger.
</td>
</tr>

</table>


See Python documentation (https://docs.python.org/3/library/logging.html)
for detailed API. Below is only a summary.

The logger has 5 levels of logging from the most serious to the least:

1. FATAL
2. ERROR
3. WARN
4. INFO
5. DEBUG

The logger has the following methods, based on these logging levels:

1. fatal(msg, *args, **kwargs)
2. error(msg, *args, **kwargs)
3. warn(msg, *args, **kwargs)
4. info(msg, *args, **kwargs)
5. debug(msg, *args, **kwargs)

The `msg` can contain string formatting.  An example of logging at the `ERROR`
level
using string formating is:

```
>>> tf.get_logger().error("The value %d is invalid.", 3)
```

You can also specify the logging verbosity.  In this case, the
WARN level log will not be emitted:

```
>>> tf.get_logger().setLevel(ERROR)
>>> tf.get_logger().warn("This is a warning.")
```