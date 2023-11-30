description: Utility class for generating batches of temporal data.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.keras.preprocessing.sequence.TimeseriesGenerator" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__getitem__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="__iter__"/>
<meta itemprop="property" content="__len__"/>
<meta itemprop="property" content="get_config"/>
<meta itemprop="property" content="on_epoch_end"/>
<meta itemprop="property" content="to_json"/>
</div>

# tf.keras.preprocessing.sequence.TimeseriesGenerator

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/preprocessing/sequence.py#L55-L245">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Utility class for generating batches of temporal data.

Inherits From: [`Sequence`](../../../../tf/keras/utils/Sequence.md)

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.keras.preprocessing.sequence.TimeseriesGenerator(
    data,
    targets,
    length,
    sampling_rate=1,
    stride=1,
    start_index=0,
    end_index=None,
    shuffle=False,
    reverse=False,
    batch_size=128
)
</code></pre>



<!-- Placeholder for "Used in" -->

Deprecated: <a href="../../../../tf/keras/preprocessing/sequence/TimeseriesGenerator.md"><code>tf.keras.preprocessing.sequence.TimeseriesGenerator</code></a> does not
operate on tensors and is not recommended for new code. Prefer using a
<a href="../../../../tf/data/Dataset.md"><code>tf.data.Dataset</code></a> which provides a more efficient and flexible mechanism for
batching, shuffling, and windowing input. See the
[tf.data guide](https://www.tensorflow.org/guide/data) for more details.

This class takes in a sequence of data-points gathered at
equal intervals, along with time series parameters such as
stride, length of history, etc., to produce batches for
training/validation.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Arguments</h2></th></tr>

<tr>
<td>
`data`<a id="data"></a>
</td>
<td>
Indexable generator (such as list or Numpy array)
containing consecutive data points (timesteps).
The data should be at 2D, and axis 0 is expected
to be the time dimension.
</td>
</tr><tr>
<td>
`targets`<a id="targets"></a>
</td>
<td>
Targets corresponding to timesteps in `data`.
It should have same length as `data`.
</td>
</tr><tr>
<td>
`length`<a id="length"></a>
</td>
<td>
Length of the output sequences (in number of timesteps).
</td>
</tr><tr>
<td>
`sampling_rate`<a id="sampling_rate"></a>
</td>
<td>
Period between successive individual timesteps
within sequences. For rate `r`, timesteps
`data[i]`, `data[i-r]`, ... `data[i - length]`
are used for create a sample sequence.
</td>
</tr><tr>
<td>
`stride`<a id="stride"></a>
</td>
<td>
Period between successive output sequences.
For stride `s`, consecutive output samples would
be centered around `data[i]`, `data[i+s]`, `data[i+2*s]`, etc.
</td>
</tr><tr>
<td>
`start_index`<a id="start_index"></a>
</td>
<td>
Data points earlier than `start_index` will not be used
in the output sequences. This is useful to reserve part of the
data for test or validation.
</td>
</tr><tr>
<td>
`end_index`<a id="end_index"></a>
</td>
<td>
Data points later than `end_index` will not be used
in the output sequences. This is useful to reserve part of the
data for test or validation.
</td>
</tr><tr>
<td>
`shuffle`<a id="shuffle"></a>
</td>
<td>
Whether to shuffle output samples,
or instead draw them in chronological order.
</td>
</tr><tr>
<td>
`reverse`<a id="reverse"></a>
</td>
<td>
Boolean: if `true`, timesteps in each output sample will be
in reverse chronological order.
</td>
</tr><tr>
<td>
`batch_size`<a id="batch_size"></a>
</td>
<td>
Number of timeseries samples in each batch
(except maybe the last one).
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A [Sequence](
https://www.tensorflow.org/api_docs/python/tf/keras/utils/Sequence)
instance.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Examples</h2></th></tr>
<tr class="alt">
<td colspan="2">
```python
from keras.src.preprocessing.sequence import TimeseriesGenerator
import numpy as np
data = np.array([[i] for i in range(50)])
targets = np.array([[i] for i in range(50)])
data_gen = TimeseriesGenerator(data, targets,
                               length=10, sampling_rate=2,
                               batch_size=2)
assert len(data_gen) == 20
batch_0 = data_gen[0]
x, y = batch_0
assert np.array_equal(x,
                      np.array([[[0], [2], [4], [6], [8]],
                                [[1], [3], [5], [7], [9]]]))
assert np.array_equal(y,
                      np.array([[10], [11]]))
```
</td>
</tr>

</table>



## Methods

<h3 id="get_config"><code>get_config</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/preprocessing/sequence.py#L195-L228">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get_config()
</code></pre>

Returns the TimeseriesGenerator configuration as Python dictionary.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A Python dictionary with the TimeseriesGenerator configuration.
</td>
</tr>

</table>



<h3 id="on_epoch_end"><code>on_epoch_end</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/utils/data_utils.py#L561-L563">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>on_epoch_end()
</code></pre>

Method called at the end of every epoch.


<h3 id="to_json"><code>to_json</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/preprocessing/sequence.py#L230-L245">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>to_json(
    **kwargs
)
</code></pre>

Returns a JSON string containing the generator's configuration.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`**kwargs`
</td>
<td>
Additional keyword arguments to be passed
to `json.dumps()`.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A JSON string containing the tokenizer configuration.
</td>
</tr>

</table>



<h3 id="__getitem__"><code>__getitem__</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/preprocessing/sequence.py#L170-L193">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__getitem__(
    index
)
</code></pre>

Gets batch at position `index`.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`index`
</td>
<td>
position of the batch in the Sequence.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A batch
</td>
</tr>

</table>



<h3 id="__iter__"><code>__iter__</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/utils/data_utils.py#L565-L568">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__iter__()
</code></pre>

Create a generator that iterate over the Sequence.


<h3 id="__len__"><code>__len__</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/preprocessing/sequence.py#L165-L168">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>__len__()
</code></pre>

Number of batch in the Sequence.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
The number of batches in the Sequence.
</td>
</tr>

</table>





