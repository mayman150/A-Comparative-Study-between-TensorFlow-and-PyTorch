<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.gfile.Remove" />
<meta itemprop="path" content="Stable" />
</div>

# tf.gfile.Remove

Deletes the file located at 'filename'.

### Aliases:

* `tf.compat.v1.gfile.Remove`
* `tf.compat.v2.compat.v1.gfile.Remove`
* `tf.gfile.Remove`

``` python
tf.gfile.Remove(filename)
```

<!-- Placeholder for "Used in" -->


#### Args:


* <b>`filename`</b>: string, a filename


#### Raises:


* <b>`errors.OpError`</b>: Propagates any errors reported by the FileSystem API.  E.g.,
NotFoundError if the file does not exist.