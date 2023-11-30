description: Represents the type of object(s) for tf.function tracing purposes.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.types.experimental.TraceType" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__eq__"/>
<meta itemprop="property" content="cast"/>
<meta itemprop="property" content="flatten"/>
<meta itemprop="property" content="from_tensors"/>
<meta itemprop="property" content="is_subtype_of"/>
<meta itemprop="property" content="most_specific_common_supertype"/>
<meta itemprop="property" content="placeholder_value"/>
<meta itemprop="property" content="to_tensors"/>
</div>

# tf.types.experimental.TraceType

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/types/trace.py">View source</a>



Represents the type of object(s) for tf.function tracing purposes.

<!-- Placeholder for "Used in" -->

`TraceType` is an abstract class that other classes might inherit from to
provide information regarding associated class(es) for the purposes of
tf.function tracing. The typing logic provided through this mechanism will be
used to make decisions regarding usage of cached concrete functions and
retracing.

For example, if we have the following tf.function and classes:
```python
@tf.function
def get_mixed_flavor(fruit_a, fruit_b):
  return fruit_a.flavor + fruit_b.flavor

class Fruit:
  flavor = tf.constant([0, 0])

class Apple(Fruit):
  flavor = tf.constant([1, 2])

class Mango(Fruit):
  flavor = tf.constant([3, 4])
```

tf.function does not know when to re-use an existing concrete function in
regards to the `Fruit` class so naively it retraces for every new instance.
```python
get_mixed_flavor(Apple(), Mango()) # Traces a new concrete function
get_mixed_flavor(Apple(), Mango()) # Traces a new concrete function again
```

However, we, as the designers of the `Fruit` class, know that each subclass
has a fixed flavor and we can reuse an existing traced concrete function if
it was the same subclass. Avoiding such unnecessary tracing of concrete
functions can have significant performance benefits.

```python
class FruitTraceType(tf.types.experimental.TraceType):
  def __init__(self, fruit):
    self.fruit_type = type(fruit)
    self.fruit_value = fruit

  def is_subtype_of(self, other):
     return (type(other) is FruitTraceType and
             self.fruit_type is other.fruit_type)

  def most_specific_common_supertype(self, others):
     return self if all(self == other for other in others) else None

  def placeholder_value(self, placeholder_context=None):
    return self.fruit_value

class Fruit:

 def __tf_tracing_type__(self, context):
   return FruitTraceType(self)
```

Now if we try calling it again:
```python
get_mixed_flavor(Apple(), Mango()) # Traces a new concrete function
get_mixed_flavor(Apple(), Mango()) # Re-uses the traced concrete function
```

## Methods

<h3 id="cast"><code>cast</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/types/trace.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>cast(
    value, cast_context
) -> Any
</code></pre>

Cast value to this type.


<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`value`
</td>
<td>
An input value belonging to this TraceType.
</td>
</tr><tr>
<td>
`cast_context`
</td>
<td>
A context reserved for internal/future usage.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
The value casted to this TraceType.
</td>
</tr>

</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Raises</th></tr>

<tr>
<td>
`AssertionError`
</td>
<td>
When _cast is not overloaded in subclass,
the value is returned directly, and it should be the same to
self.placeholder_value().
</td>
</tr>
</table>



<h3 id="flatten"><code>flatten</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/types/trace.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>flatten() -> List['TraceType']
</code></pre>

Returns a list of TensorSpecs corresponding to `to_tensors` values.


<h3 id="from_tensors"><code>from_tensors</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/types/trace.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>from_tensors(
    tensors: Iterator[core.Tensor]
) -> Any
</code></pre>

Generates a value of this type from Tensors.

Must use the same fixed amount of tensors as `to_tensors`.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`tensors`
</td>
<td>
An iterator from which the tensors can be pulled.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
A value of this type.
</td>
</tr>

</table>



<h3 id="is_subtype_of"><code>is_subtype_of</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/types/trace.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@abc.abstractmethod</code>
<code>is_subtype_of(
    other: 'TraceType'
) -> bool
</code></pre>

Returns True if `self` is a subtype of `other`.

For example, <a href="../../../tf/function.md"><code>tf.function</code></a> uses subtyping for dispatch:
if `a.is_subtype_of(b)` is True, then an argument of `TraceType`
`a` can be used as argument to a `ConcreteFunction` traced with an
a `TraceType` `b`.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`other`
</td>
<td>
A TraceType object to be compared against.
</td>
</tr>
</table>



#### Example:



```python
class Dimension(TraceType):
  def __init__(self, value: Optional[int]):
    self.value = value

  def is_subtype_of(self, other):
    # Either the value is the same or other has a generalized value that
    # can represent any specific ones.
    return (self.value == other.value) or (other.value is None)
```

<h3 id="most_specific_common_supertype"><code>most_specific_common_supertype</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/types/trace.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@abc.abstractmethod</code>
<code>most_specific_common_supertype(
    others: Sequence['TraceType']
) -> Optional['TraceType']
</code></pre>

Returns the most specific supertype of `self` and `others`, if exists.

The returned `TraceType` is a supertype of `self` and `others`, that is,
they are all subtypes (see `is_subtype_of`) of it.
It is also most specific, that is, there it has no subtype that is also
a common supertype of `self` and `others`.

If `self` and `others` have no common supertype, this returns `None`.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`others`
</td>
<td>
A sequence of TraceTypes.
</td>
</tr>
</table>



#### Example:


```python
 class Dimension(TraceType):
   def __init__(self, value: Optional[int]):
     self.value = value

   def most_specific_common_supertype(self, other):
      # Either the value is the same or other has a generalized value that
      # can represent any specific ones.
      if self.value == other.value:
        return self.value
      else:
        return Dimension(None)
```

<h3 id="placeholder_value"><code>placeholder_value</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/types/trace.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@abc.abstractmethod</code>
<code>placeholder_value(
    placeholder_context
) -> Any
</code></pre>

Creates a placeholder for tracing.

tf.funcion traces with the placeholder value rather than the actual value.
For example, a placeholder value can represent multiple different
actual values. This means that the trace generated with that placeholder
value is more general and reusable which saves expensive retracing.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`placeholder_context`
</td>
<td>
A context reserved for internal/future usage.
</td>
</tr>
</table>


For the `Fruit` example shared above, implementing:

```python
class FruitTraceType:
  def placeholder_value(self, placeholder_context):
    return Fruit()
```
instructs tf.function to trace with the `Fruit()` objects
instead of the actual `Apple()` and `Mango()` objects when it receives a
call to `get_mixed_flavor(Apple(), Mango())`. For example, Tensor arguments
are replaced with Tensors of similar shape and dtype, output from
a tf.Placeholder op.

More generally, placeholder values are the arguments of a tf.function,
as seen from the function's body:
```python
@tf.function
def foo(x):
  # Here `x` is be the placeholder value
  ...

foo(x) # Here `x` is the actual value
```

<h3 id="to_tensors"><code>to_tensors</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/types/trace.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>to_tensors(
    value: Any
) -> List[core.Tensor]
</code></pre>

Breaks down a value of this type into Tensors.

For a TraceType instance, the number of tensors generated for corresponding
value should be constant.

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Args</th></tr>

<tr>
<td>
`value`
</td>
<td>
A value belonging to this TraceType
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2">Returns</th></tr>
<tr class="alt">
<td colspan="2">
List of Tensors.
</td>
</tr>

</table>



<h3 id="__eq__"><code>__eq__</code></h3>

<a target="_blank" class="external" href="/code/stable/tensorflow/python/types/trace.py">View source</a>

<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>@abc.abstractmethod</code>
<code>__eq__(
    other
) -> bool
</code></pre>

Return self==value.




