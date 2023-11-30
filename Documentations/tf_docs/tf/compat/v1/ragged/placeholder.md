description: Creates a placeholder for a <a href="../../../../tf/RaggedTensor.md"><code>tf.RaggedTensor</code></a> that will always be fed.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.compat.v1.ragged.placeholder" />
<meta itemprop="path" content="Stable" />
</div>

# tf.compat.v1.ragged.placeholder

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/ops/ragged/ragged_factory_ops.py">View source</a>



Creates a placeholder for a <a href="../../../../tf/RaggedTensor.md"><code>tf.RaggedTensor</code></a> that will always be fed.


<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.compat.v1.ragged.placeholder(
    dtype, ragged_rank, value_shape=None, name=None
)
</code></pre>





 <section><devsite-expandable expanded>
 <h2 class="showalways">Migrate to TF2</h2>

Caution: This API was designed for TensorFlow v1.
Continue reading for details on how to migrate from this API to a native
TensorFlow v2 equivalent. See the
[TensorFlow v1 to TensorFlow v2 migration guide](https://www.tensorflow.org/guide/migrate)
for instructions on how to migrate the rest of your code.

This API is not compatible with eager execution and <a href="../../../../tf/function.md"><code>tf.function</code></a>. To migrate
to TF2, rewrite the code to be compatible with eager execution. Check the
[migration
guide](https://www.tensorflow.org/guide/migrate#1_replace_v1sessionrun_calls)
on replacing `Session.run` calls. In TF2, you can just pass tensors directly
into ops and layers. If you want to explicitly set up your inputs, also see
[Keras functional API](https://www.tensorflow.org/guide/keras/functional) on
how to use <a href="../../../../tf/keras/Input.md"><code>tf.keras.Input</code></a> to replace <a href="../../../../tf/compat/v1/ragged/placeholder.md"><code>tf.compat.v1.ragged.placeholder</code></a>.
<a href="../../../../tf/function.md"><code>tf.function</code></a> arguments also do the job of <a href="../../../../tf/compat/v1/ragged/placeholder.md"><code>tf.compat.v1.ragged.placeholder</code></a>.
For more details please read [Better
performance with tf.function](https://www.tensorflow.org/guide/function).

 </aside></devsite-expandable></section>

<h2>Description</h2>

<!-- Placeholder for "Used in" -->

**Important**: This ragged tensor will produce an error if evaluated.
Its value must be fed using the `feed_dict` optional argument to
`Session.run()`, `Tensor.eval()`, or `Operation.run()`.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`dtype`<a id="dtype"></a>
</td>
<td>
The data type for the `RaggedTensor`.
</td>
</tr><tr>
<td>
`ragged_rank`<a id="ragged_rank"></a>
</td>
<td>
The ragged rank for the `RaggedTensor`
</td>
</tr><tr>
<td>
`value_shape`<a id="value_shape"></a>
</td>
<td>
The shape for individual flat values in the `RaggedTensor`.
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
A `RaggedTensor` that may be used as a handle for feeding a value, but
not evaluated directly.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Raises</h2></th></tr>

<tr>
<td>
`RuntimeError`<a id="RuntimeError"></a>
</td>
<td>
if eager execution is enabled
</td>
</tr>
</table>


