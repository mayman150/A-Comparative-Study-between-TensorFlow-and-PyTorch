description: Computes CTC (Connectionist Temporal Classification) loss.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.nn.ctc_loss" />
<meta itemprop="path" content="Stable" />
</div>

# tf.nn.ctc_loss

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/ops/ctc_ops.py">View source</a>



Computes CTC (Connectionist Temporal Classification) loss.


<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.nn.ctc_loss(
    labels,
    logits,
    label_length,
    logit_length,
    logits_time_major=True,
    unique=None,
    blank_index=None,
    name=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

This op implements the CTC loss as presented in
[Graves et al., 2006](https://www.cs.toronto.edu/~graves/icml_2006.pdf)

Connectionist temporal classification (CTC) is a type of neural network output
and associated scoring function, for training recurrent neural networks (RNNs)
such as LSTM networks to tackle sequence problems where the timing is
variable. It can be used for tasks like on-line handwriting recognition or
recognizing phones in speech audio. CTC refers to the outputs and scoring, and
is independent of the underlying neural network structure.

#### Notes:



- This class performs the softmax operation for you, so `logits` should be
  e.g. linear projections of outputs by an LSTM.
- Outputs true repeated classes with blanks in between, and can also output
  repeated classes with no blanks in between that need to be collapsed by the
  decoder.
- `labels` may be supplied as either a dense, zero-padded `Tensor` with a
  vector of label sequence lengths OR as a `SparseTensor`.
- On TPU: Only dense padded `labels` are supported.
- On CPU and GPU: Caller may use `SparseTensor` or dense padded `labels`
  but calling with a `SparseTensor` will be significantly faster.
- Default blank label is `0` instead of `num_labels - 1` (where `num_labels`
  is the innermost dimension size of `logits`), unless overridden by
  `blank_index`.

```
>>> tf.random.set_seed(50)
>>> batch_size = 8
>>> num_labels = 6
>>> max_label_length = 5
>>> num_frames = 12
>>> labels = tf.random.uniform([batch_size, max_label_length],
...                            minval=1, maxval=num_labels, dtype=tf.int64)
>>> logits = tf.random.uniform([num_frames, batch_size, num_labels])
>>> label_length = tf.random.uniform([batch_size], minval=2,
...                                  maxval=max_label_length, dtype=tf.int64)
>>> label_mask = tf.sequence_mask(label_length, maxlen=max_label_length,
...                               dtype=label_length.dtype)
>>> labels *= label_mask
>>> logit_length = [num_frames] * batch_size
>>> with tf.GradientTape() as t:
...   t.watch(logits)
...   ref_loss = tf.nn.ctc_loss(
...       labels=labels,
...       logits=logits,
...       label_length=label_length,
...       logit_length=logit_length,
...       blank_index=0)
>>> ref_grad = t.gradient(ref_loss, logits)
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`labels`<a id="labels"></a>
</td>
<td>
`Tensor` of shape `[batch_size, max_label_seq_length]` or
`SparseTensor`.
</td>
</tr><tr>
<td>
`logits`<a id="logits"></a>
</td>
<td>
`Tensor` of shape `[frames, batch_size, num_labels]`. If
`logits_time_major == False`, shape is `[batch_size, frames, num_labels]`.
</td>
</tr><tr>
<td>
`label_length`<a id="label_length"></a>
</td>
<td>
`Tensor` of shape `[batch_size]`. None, if `labels` is a
`SparseTensor`. Length of reference label sequence in `labels`.
</td>
</tr><tr>
<td>
`logit_length`<a id="logit_length"></a>
</td>
<td>
`Tensor` of shape `[batch_size]`. Length of input sequence in
`logits`.
</td>
</tr><tr>
<td>
`logits_time_major`<a id="logits_time_major"></a>
</td>
<td>
(optional) If True (default), `logits` is shaped [frames,
batch_size, num_labels]. If False, shape is
`[batch_size, frames, num_labels]`.
</td>
</tr><tr>
<td>
`unique`<a id="unique"></a>
</td>
<td>
(optional) Unique label indices as computed by
`ctc_unique_labels(labels)`.  If supplied, enable a faster, memory
efficient implementation on TPU.
</td>
</tr><tr>
<td>
`blank_index`<a id="blank_index"></a>
</td>
<td>
(optional) Set the class index to use for the blank label.
Negative values will start from `num_labels`, ie, `-1` will reproduce the
ctc_loss behavior of using `num_labels - 1` for the blank symbol. There is
some memory/performance overhead to switching from the default of 0 as an
additional shifted copy of `logits` may be created.
</td>
</tr><tr>
<td>
`name`<a id="name"></a>
</td>
<td>
A name for this `Op`. Defaults to "ctc_loss_dense".
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>

<tr>
<td>
`loss`<a id="loss"></a>
</td>
<td>
A 1-D `float Tensor` of shape `[batch_size]`, containing negative log
probabilities.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Raises</h2></th></tr>

<tr>
<td>
`ValueError`<a id="ValueError"></a>
</td>
<td>
Argument `blank_index` must be provided when `labels` is a
`SparseTensor`.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">References</h2></th></tr>
<tr class="alt">
<td colspan="2">
Connectionist Temporal Classification - Labeling Unsegmented Sequence Data
with Recurrent Neural Networks:
  [Graves et al., 2006](https://dl.acm.org/citation.cfm?id=1143891)
  ([pdf](http://www.cs.toronto.edu/~graves/icml_2006.pdf))

https://en.wikipedia.org/wiki/Connectionist_temporal_classification
</td>
</tr>

</table>

