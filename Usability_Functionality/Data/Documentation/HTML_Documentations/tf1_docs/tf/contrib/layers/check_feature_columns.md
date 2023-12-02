<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.contrib.layers.check_feature_columns" />
<meta itemprop="path" content="Stable" />
</div>

# tf.contrib.layers.check_feature_columns

Checks the validity of the set of FeatureColumns.

``` python
tf.contrib.layers.check_feature_columns(feature_columns)
```

<!-- Placeholder for "Used in" -->


#### Args:


* <b>`feature_columns`</b>: An iterable of instances or subclasses of FeatureColumn.


#### Raises:


* <b>`ValueError`</b>: If `feature_columns` is a dict.
* <b>`ValueError`</b>: If there are duplicate feature column keys.