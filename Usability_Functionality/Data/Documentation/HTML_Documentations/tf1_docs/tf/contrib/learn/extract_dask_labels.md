<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.contrib.learn.extract_dask_labels" />
<meta itemprop="path" content="Stable" />
</div>

# tf.contrib.learn.extract_dask_labels

Extract data from dask.Series or dask.DataFrame for labels. (deprecated)

``` python
tf.contrib.learn.extract_dask_labels(labels)
```

<!-- Placeholder for "Used in" -->

Warning: THIS FUNCTION IS DEPRECATED. It will be removed in a future version.
Instructions for updating:
Please feed input to tf.data to support dask.

Given a distributed dask.DataFrame or dask.Series containing exactly one
column or name, this operation returns a single dask.DataFrame or dask.Series
that can be iterated over.

#### Args:


* <b>`labels`</b>: A distributed dask.DataFrame or dask.Series with exactly one
        column or name.


#### Returns:

A dask.DataFrame or dask.Series that can be iterated over.
If the supplied argument is neither a dask.DataFrame nor a dask.Series this
operation returns it without modification.



#### Raises:


* <b>`ValueError`</b>: If the supplied dask.DataFrame contains more than one
            column or the supplied dask.Series contains more than
            one name.