description: Builds a Enqueuer from a Sequence.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.keras.utils.OrderedEnqueuer" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="get"/>
<meta itemprop="property" content="is_running"/>
<meta itemprop="property" content="start"/>
<meta itemprop="property" content="stop"/>
</div>

# tf.keras.utils.OrderedEnqueuer

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/utils/data_utils.py#L779-L871">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Builds a Enqueuer from a Sequence.

Inherits From: [`SequenceEnqueuer`](../../../tf/keras/utils/SequenceEnqueuer.md)

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.keras.utils.OrderedEnqueuer(
    sequence, use_multiprocessing=False, shuffle=False
)
</code></pre>



<!-- Placeholder for "Used in" -->


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`sequence`<a id="sequence"></a>
</td>
<td>
A `tf.keras.utils.data_utils.Sequence` object.
</td>
</tr><tr>
<td>
`use_multiprocessing`<a id="use_multiprocessing"></a>
</td>
<td>
use multiprocessing if True, otherwise threading
</td>
</tr><tr>
<td>
`shuffle`<a id="shuffle"></a>
</td>
<td>
whether to shuffle the data at the beginning of each epoch
</td>
</tr>
</table>



## Methods

<h3 id="get"><code>get</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/utils/data_utils.py#L850-L871">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>get()
</code></pre>

Creates a generator to extract data from the queue.

Skip the data if it is `None`.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Yields</th></tr>
<tr class="alt">
<td colspan="2">
The next element in the queue, i.e. a tuple
`(inputs, targets)` or
`(inputs, targets, sample_weights)`.
</td>
</tr>

</table>



<h3 id="is_running"><code>is_running</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/utils/data_utils.py#L702-L703">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>is_running()
</code></pre>




<h3 id="start"><code>start</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/utils/data_utils.py#L705-L723">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>start(
    workers=1, max_queue_size=10
)
</code></pre>

Starts the handler's workers.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`workers`
</td>
<td>
Number of workers.
</td>
</tr><tr>
<td>
`max_queue_size`
</td>
<td>
queue size
(when full, workers could block on `put()`)
</td>
</tr>
</table>



<h3 id="stop"><code>stop</code></h3>

<a target="_blank" class="external" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/utils/data_utils.py#L730-L744">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>stop(
    timeout=None
)
</code></pre>

Stops running threads and wait for them to exit, if necessary.

Should be called by the same thread which called `start()`.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`timeout`
</td>
<td>
maximum time to wait on `thread.join()`
</td>
</tr>
</table>





