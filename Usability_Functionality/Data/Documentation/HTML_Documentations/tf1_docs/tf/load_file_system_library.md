<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.load_file_system_library" />
<meta itemprop="path" content="Stable" />
</div>

# tf.load_file_system_library

Loads a TensorFlow plugin, containing file system implementation. (deprecated)

### Aliases:

* `tf.compat.v1.load_file_system_library`
* `tf.compat.v2.compat.v1.load_file_system_library`
* `tf.load_file_system_library`

``` python
tf.load_file_system_library(library_filename)
```

<!-- Placeholder for "Used in" -->

Warning: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating:
Use <a href="../tf/load_library.md"><code>tf.load_library</code></a> instead.

Pass `library_filename` to a platform-specific mechanism for dynamically
loading a library. The rules for determining the exact location of the
library are platform-specific and are not documented here.

#### Args:


* <b>`library_filename`</b>: Path to the plugin.
  Relative or absolute filesystem path to a dynamic library file.


#### Returns:

None.



#### Raises:


* <b>`RuntimeError`</b>: when unable to load the library.