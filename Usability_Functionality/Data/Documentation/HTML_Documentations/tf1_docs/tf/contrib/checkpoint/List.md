<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.contrib.checkpoint.List" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="layers"/>
<meta itemprop="property" content="losses"/>
<meta itemprop="property" content="non_trainable_variables"/>
<meta itemprop="property" content="non_trainable_weights"/>
<meta itemprop="property" content="trainable"/>
<meta itemprop="property" content="trainable_variables"/>
<meta itemprop="property" content="trainable_weights"/>
<meta itemprop="property" content="updates"/>
<meta itemprop="property" content="variables"/>
<meta itemprop="property" content="weights"/>
<meta itemprop="property" content="__add__"/>
<meta itemprop="property" content="__contains__"/>
<meta itemprop="property" content="__eq__"/>
<meta itemprop="property" content="__getitem__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="__iter__"/>
<meta itemprop="property" content="__len__"/>
<meta itemprop="property" content="__mul__"/>
<meta itemprop="property" content="__radd__"/>
<meta itemprop="property" content="__rmul__"/>
<meta itemprop="property" content="append"/>
<meta itemprop="property" content="copy"/>
<meta itemprop="property" content="count"/>
<meta itemprop="property" content="extend"/>
<meta itemprop="property" content="index"/>
</div>

# tf.contrib.checkpoint.List

## Class `List`

An append-only sequence type which is trackable.



<!-- Placeholder for "Used in" -->

Maintains checkpoint dependencies on its contents (which must also be
trackable), and forwards any `Layer` metadata such as updates and losses.

Note that `List` is purely a container. It lets a <a href="../../../tf/keras/Model.md"><code>tf.keras.Model</code></a> or
other trackable object know about its contents, but does not call any
`Layer` instances which are added to it. To indicate a sequence of `Layer`
instances which should be called sequentially, use <a href="../../../tf/keras/Sequential.md"><code>tf.keras.Sequential</code></a>.

#### Example usage:


```python
class HasList(tf.keras.Model):

  def __init__(self):
    super(HasList, self).__init__()
    self.layer_list = tf.contrib.checkpoint.List([layers.Dense(3)])
    self.layer_list.append(layers.Dense(4))

  def call(self, x):
    aggregation = 0.
    for l in self.layer_list:
      x = l(x)
      aggregation += tf.reduce_sum(x)
    return aggregation
```

This kind of wrapping is necessary because `Trackable` objects do not
(yet) deeply inspect regular Python data structures, so for example assigning
a regular list (`self.layer_list = [layers.Dense(3)]`) does not create a
checkpoint dependency and does not add the `Layer` instance's weights to its
parent `Model`.

<h2 id="__init__"><code>__init__</code></h2>

``` python
__init__(
    *args,
    **kwargs
)
```

Construct a new sequence. Arguments are passed to `list()`.




## Properties

<h3 id="layers"><code>layers</code></h3>




<h3 id="losses"><code>losses</code></h3>

Aggregate losses from any `Layer` instances.


<h3 id="non_trainable_variables"><code>non_trainable_variables</code></h3>




<h3 id="non_trainable_weights"><code>non_trainable_weights</code></h3>




<h3 id="trainable"><code>trainable</code></h3>




<h3 id="trainable_variables"><code>trainable_variables</code></h3>




<h3 id="trainable_weights"><code>trainable_weights</code></h3>




<h3 id="updates"><code>updates</code></h3>

Aggregate updates from any `Layer` instances.


<h3 id="variables"><code>variables</code></h3>




<h3 id="weights"><code>weights</code></h3>






## Methods

<h3 id="__add__"><code>__add__</code></h3>

``` python
__add__(other)
```




<h3 id="__contains__"><code>__contains__</code></h3>

``` python
__contains__(value)
```




<h3 id="__eq__"><code>__eq__</code></h3>

``` python
__eq__(other)
```

Return self==value.


<h3 id="__getitem__"><code>__getitem__</code></h3>

``` python
__getitem__(key)
```




<h3 id="__iter__"><code>__iter__</code></h3>

``` python
__iter__()
```




<h3 id="__len__"><code>__len__</code></h3>

``` python
__len__()
```




<h3 id="__mul__"><code>__mul__</code></h3>

``` python
__mul__(n)
```




<h3 id="__radd__"><code>__radd__</code></h3>

``` python
__radd__(other)
```




<h3 id="__rmul__"><code>__rmul__</code></h3>

``` python
__rmul__(n)
```




<h3 id="append"><code>append</code></h3>

``` python
append(value)
```

Add a new trackable value.


<h3 id="copy"><code>copy</code></h3>

``` python
copy()
```




<h3 id="count"><code>count</code></h3>

``` python
count(value)
```

S.count(value) -> integer -- return number of occurrences of value


<h3 id="extend"><code>extend</code></h3>

``` python
extend(values)
```

Add a sequence of trackable values.


<h3 id="index"><code>index</code></h3>

``` python
index(
    value,
    start=0,
    stop=None
)
```

S.index(value, [start, [stop]]) -> integer -- return first index of value.
Raises ValueError if the value is not present.

Supporting start and stop arguments is optional, but
recommended.



