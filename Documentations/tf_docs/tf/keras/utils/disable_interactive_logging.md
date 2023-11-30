description: Turn off interactive logging.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.keras.utils.disable_interactive_logging" />
<meta itemprop="path" content="Stable" />
</div>

# tf.keras.utils.disable_interactive_logging

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/utils/io_utils.py#L44-L52">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Turn off interactive logging.


<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.keras.utils.disable_interactive_logging()
</code></pre>



<!-- Placeholder for "Used in" -->

When interactive logging is disabled, Keras sends logs to `absl.logging`.
This is the best option when using Keras in a non-interactive
way, such as running a training or inference job on a server.