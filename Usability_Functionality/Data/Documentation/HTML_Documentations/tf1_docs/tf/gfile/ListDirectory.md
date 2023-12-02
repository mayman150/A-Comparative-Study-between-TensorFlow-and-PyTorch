<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.gfile.ListDirectory" />
<meta itemprop="path" content="Stable" />
</div>

# tf.gfile.ListDirectory

Returns a list of entries contained within a directory.

### Aliases:

* `tf.compat.v1.gfile.ListDirectory`
* `tf.compat.v2.compat.v1.gfile.ListDirectory`
* `tf.gfile.ListDirectory`

``` python
tf.gfile.ListDirectory(dirname)
```

<!-- Placeholder for "Used in" -->

The list is in arbitrary order. It does not contain the special entries "."
and "..".

#### Args:


* <b>`dirname`</b>: string, path to a directory


#### Returns:

[filename1, filename2, ... filenameN] as strings



#### Raises:

errors.NotFoundError if directory doesn't exist
