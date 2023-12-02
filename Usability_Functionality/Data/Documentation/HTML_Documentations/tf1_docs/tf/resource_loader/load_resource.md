<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.resource_loader.load_resource" />
<meta itemprop="path" content="Stable" />
</div>

# tf.resource_loader.load_resource

Load the resource at given path, where path is relative to tensorflow/.

### Aliases:

* `tf.compat.v1.resource_loader.load_resource`
* `tf.compat.v2.compat.v1.resource_loader.load_resource`
* `tf.resource_loader.load_resource`

``` python
tf.resource_loader.load_resource(path)
```

<!-- Placeholder for "Used in" -->


#### Args:


* <b>`path`</b>: a string resource path relative to tensorflow/.


#### Returns:

The contents of that resource.



#### Raises:


* <b>`IOError`</b>: If the path is not found, or the resource can't be opened.