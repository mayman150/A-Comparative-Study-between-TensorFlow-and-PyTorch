<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.gfile.Exists" />
<meta itemprop="path" content="Stable" />
</div>

# tf.gfile.Exists

Determines whether a path exists or not.

### Aliases:

* `tf.compat.v1.gfile.Exists`
* `tf.compat.v2.compat.v1.gfile.Exists`
* `tf.gfile.Exists`

``` python
tf.gfile.Exists(filename)
```

<!-- Placeholder for "Used in" -->


#### Args:


* <b>`filename`</b>: string, a path


#### Returns:

True if the path exists, whether it's a file or a directory.
False if the path does not exist and there are no filesystem errors.



#### Raises:


* <b>`errors.OpError`</b>: Propagates any errors reported by the FileSystem API.