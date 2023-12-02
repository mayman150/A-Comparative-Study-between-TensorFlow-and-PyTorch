<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.contrib.autograph.set_element_type" />
<meta itemprop="path" content="Stable" />
</div>

# tf.contrib.autograph.set_element_type

Indicates that the entity is expected hold items of specified type/shape.

``` python
tf.contrib.autograph.set_element_type(
    entity,
    dtype,
    shape=UNSPECIFIED
)
```

<!-- Placeholder for "Used in" -->

The staged TensorFlow ops will reflect and assert this data type. Ignored
otherwise.

#### Args:


* <b>`entity`</b>: The entity to annotate.
* <b>`dtype`</b>: TensorFlow dtype value to assert for entity.
* <b>`shape`</b>: Optional shape to assert for entity.