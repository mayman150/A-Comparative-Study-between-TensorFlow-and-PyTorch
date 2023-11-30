description: Wrapper allowing a stack of RNN cells to behave as a single cell.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.keras.layers.StackedRNNCells" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="get_initial_state"/>
</div>

# tf.keras.layers.StackedRNNCells

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/layers/rnn/stacked_rnn_cells.py#L34-L217">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Wrapper allowing a stack of RNN cells to behave as a single cell.

Inherits From: [`Layer`](../../../tf/keras/layers/Layer.md), [`Module`](../../../tf/Module.md)

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.keras.layers.StackedRNNCells(
    cells, **kwargs
)
</code></pre>



<!-- Placeholder for "Used in" -->

Used to implement efficient stacked RNNs.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`cells`<a id="cells"></a>
</td>
<td>
List of RNN cell instances.
</td>
</tr>
</table>



#### Examples:



```python
batch_size = 3
sentence_max_length = 5
n_features = 2
new_shape = (batch_size, sentence_max_length, n_features)
x = tf.constant(np.reshape(np.arange(30), new_shape), dtype = tf.float32)

rnn_cells = [tf.keras.layers.LSTMCell(128) for _ in range(2)]
stacked_lstm = tf.keras.layers.StackedRNNCells(rnn_cells)
lstm_layer = tf.keras.layers.RNN(stacked_lstm)

result = lstm_layer(x)
```



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Attributes</h2></th></tr>

<tr>
<td>
`output_size`<a id="output_size"></a>
</td>
<td>

</td>
</tr><tr>
<td>
`state_size`<a id="state_size"></a>
</td>
<td>

</td>
</tr>
</table>



## Methods

<h3 id="get_initial_state"><code>get_initial_state</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/layers/rnn/stacked_rnn_cells.py#L106-L125">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_initial_state(
    inputs=None, batch_size=None, dtype=None
)
</code></pre>






