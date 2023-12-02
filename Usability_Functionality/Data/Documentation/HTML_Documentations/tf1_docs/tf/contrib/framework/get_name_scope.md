<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.contrib.framework.get_name_scope" />
<meta itemprop="path" content="Stable" />
</div>

# tf.contrib.framework.get_name_scope

Returns the current name scope of the default graph.

``` python
tf.contrib.framework.get_name_scope()
```

<!-- Placeholder for "Used in" -->


#### For example:


```python
with tf.name_scope('scope1'):
  with tf.name_scope('scope2'):
    print(tf.contrib.framework.get_name_scope())
```
would print the string `scope1/scope2`.



#### Returns:

A string representing the current name scope.
