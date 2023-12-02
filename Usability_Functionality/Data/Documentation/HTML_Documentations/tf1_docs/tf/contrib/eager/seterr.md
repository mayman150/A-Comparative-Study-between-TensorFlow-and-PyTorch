<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.contrib.eager.seterr" />
<meta itemprop="path" content="Stable" />
</div>

# tf.contrib.eager.seterr

Set how abnormal conditions are handled by the default eager context.

``` python
tf.contrib.eager.seterr(inf_or_nan=None)
```

<!-- Placeholder for "Used in" -->


#### Example:


```python
tfe.seterr(inf_or_nan=ExecutionCallback.RAISE)
a = tf.constant(10.0)
b = tf.constant(0.0)
try:
  c = a / b  # <-- Raises InfOrNanError.
except Exception as e:
  print("Caught Exception: %s" % e)

tfe.seterr(inf_or_nan=ExecutionCallback.IGNORE)
c = a / b  # <-- Does NOT raise exception anymore.
```

#### Args:


* <b>`inf_or_nan`</b>: An `ExecutionCallback` determining the action for infinity
  (`inf`) and NaN (`nan`) values. A value of `None` leads to no change in
  the action of the condition.


#### Returns:

A dictionary of old actions.



#### Raises:


* <b>`ValueError`</b>: If the value of any keyword arguments is invalid.