<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.contrib.framework.zero_initializer" />
<meta itemprop="path" content="Stable" />
</div>

# tf.contrib.framework.zero_initializer

Initialize 'ref' with all zeros, ref tensor should be uninitialized.

``` python
tf.contrib.framework.zero_initializer(
    ref,
    use_locking=True,
    name='zero_initializer'
)
```

<!-- Placeholder for "Used in" -->

If already initialized, you will get ValueError. This op is intended to
save memory during initialization.
Args:
  ref: ref of the tensor need to be zero initialized.
  name: optional name for this operation.

#### Returns:

ref that initialized.


#### Raises:


* <b>`ValueError`</b>: If ref tensor is initialized.