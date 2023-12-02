<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.contrib.layers.dense_to_sparse" />
<meta itemprop="path" content="Stable" />
</div>

# tf.contrib.layers.dense_to_sparse

Converts a dense tensor into a sparse tensor.

``` python
tf.contrib.layers.dense_to_sparse(
    tensor,
    eos_token=0,
    outputs_collections=None,
    scope=None
)
```

<!-- Placeholder for "Used in" -->

An example use would be to convert dense labels to sparse ones
so that they can be fed to the ctc_loss.

#### Args:


* <b>`tensor`</b>: An `int` `Tensor` to be converted to a `Sparse`.
* <b>`eos_token`</b>: An integer. It is part of the target label that signifies the
  end of a sentence.
* <b>`outputs_collections`</b>: Collection to add the outputs.
* <b>`scope`</b>: Optional scope for name_scope.