<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.contrib.framework.get_variable_full_name" />
<meta itemprop="path" content="Stable" />
</div>

# tf.contrib.framework.get_variable_full_name

Returns the full name of a variable.

``` python
tf.contrib.framework.get_variable_full_name(var)
```

<!-- Placeholder for "Used in" -->

For normal Variables, this is the same as the var.op.name.  For
sliced or PartitionedVariables, this name is the same for all the
slices/partitions. In both cases, this is normally the name used in
a checkpoint file.

#### Args:


* <b>`var`</b>: A `Variable` object.


#### Returns:

A string that is the full name.
