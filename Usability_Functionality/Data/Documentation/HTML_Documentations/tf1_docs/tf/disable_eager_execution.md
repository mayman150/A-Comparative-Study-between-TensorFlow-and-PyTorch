<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.disable_eager_execution" />
<meta itemprop="path" content="Stable" />
</div>

# tf.disable_eager_execution

Disables eager execution.

### Aliases:

* `tf.compat.v1.disable_eager_execution`
* `tf.compat.v2.compat.v1.disable_eager_execution`
* `tf.disable_eager_execution`

``` python
tf.disable_eager_execution()
```

<!-- Placeholder for "Used in" -->

This function can only be called before any Graphs, Ops, or Tensors have been
created. It can be used at the beginning of the program for complex migration
projects from TensorFlow 1.x to 2.x.