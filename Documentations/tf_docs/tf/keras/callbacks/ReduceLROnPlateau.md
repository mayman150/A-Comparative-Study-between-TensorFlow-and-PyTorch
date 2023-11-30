description: Reduce learning rate when a metric has stopped improving.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.keras.callbacks.ReduceLROnPlateau" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="in_cooldown"/>
<meta itemprop="property" content="set_model"/>
<meta itemprop="property" content="set_params"/>
</div>

# tf.keras.callbacks.ReduceLROnPlateau

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/callbacks.py#L3007-L3144">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Reduce learning rate when a metric has stopped improving.

Inherits From: [`Callback`](../../../tf/keras/callbacks/Callback.md)

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.keras.callbacks.ReduceLROnPlateau(
    monitor=&#x27;val_loss&#x27;,
    factor=0.1,
    patience=10,
    verbose=0,
    mode=&#x27;auto&#x27;,
    min_delta=0.0001,
    cooldown=0,
    min_lr=0,
    **kwargs
)
</code></pre>



<!-- Placeholder for "Used in" -->

Models often benefit from reducing the learning rate by a factor
of 2-10 once learning stagnates. This callback monitors a
quantity and if no improvement is seen for a 'patience' number
of epochs, the learning rate is reduced.

#### Example:



```python
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=5, min_lr=0.001)
model.fit(X_train, Y_train, callbacks=[reduce_lr])
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`monitor`<a id="monitor"></a>
</td>
<td>
quantity to be monitored.
</td>
</tr><tr>
<td>
`factor`<a id="factor"></a>
</td>
<td>
factor by which the learning rate will be reduced.
`new_lr = lr * factor`.
</td>
</tr><tr>
<td>
`patience`<a id="patience"></a>
</td>
<td>
number of epochs with no improvement after which learning rate
will be reduced.
</td>
</tr><tr>
<td>
`verbose`<a id="verbose"></a>
</td>
<td>
int. 0: quiet, 1: update messages.
</td>
</tr><tr>
<td>
`mode`<a id="mode"></a>
</td>
<td>
one of `{'auto', 'min', 'max'}`. In `'min'` mode,
the learning rate will be reduced when the
quantity monitored has stopped decreasing; in `'max'` mode it will be
reduced when the quantity monitored has stopped increasing; in
`'auto'` mode, the direction is automatically inferred from the name
of the monitored quantity.
</td>
</tr><tr>
<td>
`min_delta`<a id="min_delta"></a>
</td>
<td>
threshold for measuring the new optimum, to only focus on
significant changes.
</td>
</tr><tr>
<td>
`cooldown`<a id="cooldown"></a>
</td>
<td>
number of epochs to wait before resuming normal operation
after lr has been reduced.
</td>
</tr><tr>
<td>
`min_lr`<a id="min_lr"></a>
</td>
<td>
lower bound on the learning rate.
</td>
</tr>
</table>



## Methods

<h3 id="in_cooldown"><code>in_cooldown</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/callbacks.py#L3143-L3144">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>in_cooldown()
</code></pre>




<h3 id="set_model"><code>set_model</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/callbacks.py#L694-L695">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>set_model(
    model
)
</code></pre>




<h3 id="set_params"><code>set_params</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/callbacks.py#L691-L692">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>set_params(
    params
)
</code></pre>






