<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.contrib.eager.nan_callback" />
<meta itemprop="path" content="Stable" />
</div>

# tf.contrib.eager.nan_callback

A specialization of `inf_nan_callback` that checks for `nan`s only.

``` python
tf.contrib.eager.nan_callback(
    op_type,
    inputs,
    attrs,
    outputs,
    op_name,
    action=tf.contrib.eager.ExecutionCallback.RAISE
)
```

<!-- Placeholder for "Used in" -->
