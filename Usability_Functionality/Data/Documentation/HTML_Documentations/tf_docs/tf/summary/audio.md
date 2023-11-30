description: Write an audio summary.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.summary.audio" />
<meta itemprop="path" content="Stable" />
</div>

# tf.summary.audio

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/tensorflow/tensorboard/tree/2.15.1/tensorboard/plugins/audio/summary_v2.py#L32-L125">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Write an audio summary.


<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.summary.audio(
    name,
    data,
    sample_rate,
    step=None,
    max_outputs=3,
    encoding=None,
    description=None
)
</code></pre>



<!-- Placeholder for "Used in" -->


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Arguments</h2></th></tr>

<tr>
<td>
`name`<a id="name"></a>
</td>
<td>
A name for this summary. The summary tag used for TensorBoard will
be this name prefixed by any active name scopes.
</td>
</tr><tr>
<td>
`data`<a id="data"></a>
</td>
<td>
A `Tensor` representing audio data with shape `[k, t, c]`,
where `k` is the number of audio clips, `t` is the number of
frames, and `c` is the number of channels. Elements should be
floating-point values in `[-1.0, 1.0]`. Any of the dimensions may
be statically unknown (i.e., `None`).
</td>
</tr><tr>
<td>
`sample_rate`<a id="sample_rate"></a>
</td>
<td>
An `int` or rank-0 `int32` `Tensor` that represents the
sample rate, in Hz. Must be positive.
</td>
</tr><tr>
<td>
`step`<a id="step"></a>
</td>
<td>
Explicit `int64`-castable monotonic step value for this summary. If
omitted, this defaults to `tf.summary.experimental.get_step()`, which must
not be None.
</td>
</tr><tr>
<td>
`max_outputs`<a id="max_outputs"></a>
</td>
<td>
Optional `int` or rank-0 integer `Tensor`. At most this
many audio clips will be emitted at each step. When more than
`max_outputs` many clips are provided, the first `max_outputs`
many clips will be used and the rest silently discarded.
</td>
</tr><tr>
<td>
`encoding`<a id="encoding"></a>
</td>
<td>
Optional constant `str` for the desired encoding. Only "wav"
is currently supported, but this is not guaranteed to remain the
default, so if you want "wav" in particular, set this explicitly.
</td>
</tr><tr>
<td>
`description`<a id="description"></a>
</td>
<td>
Optional long-form description for this summary, as a
constant `str`. Markdown is supported. Defaults to empty.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
True on success, or false if no summary was emitted because no default
summary writer was available.
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
if a default writer exists, but no step was provided and
`tf.summary.experimental.get_step()` is None.
</td>
</tr>
</table>

