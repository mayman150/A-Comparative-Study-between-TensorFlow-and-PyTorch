description: Joins the elements of inputs based on segment_ids.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.strings.unsorted_segment_join" />
<meta itemprop="path" content="Stable" />
</div>

# tf.strings.unsorted_segment_join

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/ops/string_ops.py">View source</a>



Joins the elements of `inputs` based on `segment_ids`.


<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.strings.unsorted_segment_join`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.strings.unsorted_segment_join(
    inputs, segment_ids, num_segments, separator=&#x27;&#x27;, name=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

Computes the string join along segments of a tensor.

Given `segment_ids` with rank `N` and `data` with rank `N+M`:

```
output[i, k1...kM] = strings.join([data[j1...jN, k1...kM])
```

where the join is over all `[j1...jN]` such that `segment_ids[j1...jN] = i`.

Strings are joined in row-major order.

#### For example:



```
>>> inputs = ['this', 'a', 'test', 'is']
>>> segment_ids = [0, 1, 1, 0]
>>> num_segments = 2
>>> separator = ' '
>>> tf.strings.unsorted_segment_join(inputs, segment_ids, num_segments,
...                                  separator).numpy()
array([b'this is', b'a test'], dtype=object)
```

```
>>> inputs = [['Y', 'q', 'c'], ['Y', '6', '6'], ['p', 'G', 'a']]
>>> segment_ids = [1, 0, 1]
>>> num_segments = 2
>>> tf.strings.unsorted_segment_join(inputs, segment_ids, num_segments,
...                                  separator=':').numpy()
array([[b'Y', b'6', b'6'],
       [b'Y:p', b'q:G', b'c:a']], dtype=object)
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`inputs`<a id="inputs"></a>
</td>
<td>
A list of <a href="../../tf/Tensor.md"><code>tf.Tensor</code></a> objects of type <a href="../../tf.md#string"><code>tf.string</code></a>.
</td>
</tr><tr>
<td>
`segment_ids`<a id="segment_ids"></a>
</td>
<td>
A tensor whose shape is a prefix of `inputs.shape` and whose
type must be <a href="../../tf.md#int32"><code>tf.int32</code></a> or <a href="../../tf.md#int64"><code>tf.int64</code></a>. Negative segment ids are not
supported.
</td>
</tr><tr>
<td>
`num_segments`<a id="num_segments"></a>
</td>
<td>
A scalar of type <a href="../../tf.md#int32"><code>tf.int32</code></a> or <a href="../../tf.md#int64"><code>tf.int64</code></a>. Must be
non-negative and larger than any segment id.
</td>
</tr><tr>
<td>
`separator`<a id="separator"></a>
</td>
<td>
The separator to use when joining. Defaults to `""`.
</td>
</tr><tr>
<td>
`name`<a id="name"></a>
</td>
<td>
A name for the operation (optional).
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A <a href="../../tf.md#string"><code>tf.string</code></a> tensor representing the concatenated values, using the given
separator.
</td>
</tr>

</table>

