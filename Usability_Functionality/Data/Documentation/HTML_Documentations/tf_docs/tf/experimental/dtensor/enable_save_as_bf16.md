description: Allows float32 DVariables to be checkpointed and restored as bfloat16.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.experimental.dtensor.enable_save_as_bf16" />
<meta itemprop="path" content="Stable" />
</div>

# tf.experimental.dtensor.enable_save_as_bf16

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/dtensor/python/save_restore.py">View source</a>



Allows float32 DVariables to be checkpointed and restored as bfloat16.


<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.experimental.dtensor.enable_save_as_bf16(
    variables: List[<a href="../../../tf/Variable.md"><code>tf.Variable</code></a>]
)
</code></pre>



<!-- Placeholder for "Used in" -->

The method only affects the DVariable part inside the model and leaves
non-DTensor Variables/Tensors untouched.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`variables`<a id="variables"></a>
</td>
<td>
A list of tf.Variable to be enabled with bfloat16 save/restore.
Only has effect on DTensor Variables as they go through d_variables with
DTensor Specific logis.
</td>
</tr>
</table>

