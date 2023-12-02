<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.contrib.framework.strip_name_scope" />
<meta itemprop="path" content="Stable" />
</div>

# tf.contrib.framework.strip_name_scope

Removes name scope from a name.

``` python
tf.contrib.framework.strip_name_scope(
    name,
    export_scope
)
```

<!-- Placeholder for "Used in" -->


#### Args:


* <b>`name`</b>: A `string` name.
* <b>`export_scope`</b>: Optional `string`. Name scope to remove.


#### Returns:

Name with name scope removed, or the original name if export_scope
is None.
