description: Applies a for loop.
robots: noindex

# tf.raw_ops.For

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>



Applies a for loop.


<section class="expandable">
  <h4 class="showalways">View aliases</h4>
  <p>
<b>Compat aliases for migration</b>
<p>See
<a href="https://www.tensorflow.org/guide/migrate">Migration guide</a> for
more details.</p>
<p>`tf.compat.v1.raw_ops.For`</p>
</p>
</section>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.raw_ops.For(
    start, limit, delta, input, body, name=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

  ```python
   output = input;
   for i in range(start, limit, delta)
     output = body(i, output);
  ```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`start`<a id="start"></a>
</td>
<td>
A `Tensor` of type `int32`. The lower bound. An int32
</td>
</tr><tr>
<td>
`limit`<a id="limit"></a>
</td>
<td>
A `Tensor` of type `int32`. The upper bound. An int32
</td>
</tr><tr>
<td>
`delta`<a id="delta"></a>
</td>
<td>
A `Tensor` of type `int32`. The increment. An int32
</td>
</tr><tr>
<td>
`input`<a id="input"></a>
</td>
<td>
A list of `Tensor` objects.
A list of input tensors whose types are T.
</td>
</tr><tr>
<td>
`body`<a id="body"></a>
</td>
<td>
A function decorated with @Defun.
A function that takes a list of tensors (int32, T) and returns another
list of tensors (T).
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
A list of `Tensor` objects. Has the same type as `input`.
</td>
</tr>

</table>

