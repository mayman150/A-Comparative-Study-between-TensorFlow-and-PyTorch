description: Serialize the optimizer configuration to JSON compatible python dict.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.keras.optimizers.serialize" />
<meta itemprop="path" content="Stable" />
</div>

# tf.keras.optimizers.serialize

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/optimizers/__init__.py#L74-L108">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Serialize the optimizer configuration to JSON compatible python dict.


<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.keras.optimizers.serialize(
    optimizer, use_legacy_format=False
)
</code></pre>



<!-- Placeholder for "Used in" -->

The configuration can be used for persistence and reconstruct the
`Optimizer` instance again.

```
>>> tf.keras.optimizers.serialize(tf.keras.optimizers.legacy.SGD())
{'module': 'keras.optimizers.legacy', 'class_name': 'SGD', 'config': {'name': 'SGD', 'learning_rate': 0.01, 'decay': 0.0, 'momentum': 0.0, 'nesterov': False}, 'registered_name': None}
```