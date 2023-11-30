description: Returns the tensor value of the given variable in the checkpoint.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.train.load_variable" />
<meta itemprop="path" content="Stable" />
</div>

# tf.train.load_variable

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/training/checkpoint_utils.py">View source</a>



Returns the tensor value of the given variable in the checkpoint.


<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.train.load_variable`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.train.load_variable(
    ckpt_dir_or_file, name
)
</code></pre>



<!-- Placeholder for "Used in" -->

When the variable name is unknown, you can use <a href="../../tf/train/list_variables.md"><code>tf.train.list_variables</code></a> to
inspect all the variable names.

#### Example usage:



```python
import tensorflow as tf
a = tf.Variable(1.0)
b = tf.Variable(2.0)
ckpt = tf.train.Checkpoint(var_list={'a': a, 'b': b})
ckpt_path = ckpt.save('tmp-ckpt')
var= tf.train.load_variable(
    ckpt_path, 'var_list/a/.ATTRIBUTES/VARIABLE_VALUE')
print(var)  # 1.0
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`ckpt_dir_or_file`<a id="ckpt_dir_or_file"></a>
</td>
<td>
Directory with checkpoints file or path to checkpoint.
</td>
</tr><tr>
<td>
`name`<a id="name"></a>
</td>
<td>
Name of the variable to return.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
A numpy `ndarray` with a copy of the value of this variable.
</td>
</tr>

</table>

