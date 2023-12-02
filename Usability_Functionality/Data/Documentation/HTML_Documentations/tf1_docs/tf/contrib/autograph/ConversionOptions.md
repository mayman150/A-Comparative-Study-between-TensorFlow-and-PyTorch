<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.contrib.autograph.ConversionOptions" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__eq__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="as_tuple"/>
<meta itemprop="property" content="call_options"/>
<meta itemprop="property" content="to_ast"/>
<meta itemprop="property" content="uses"/>
</div>

# tf.contrib.autograph.ConversionOptions

## Class `ConversionOptions`

Immutable container for global conversion flags.



<!-- Placeholder for "Used in" -->


#### Attributes:


* <b>`recursive`</b>: bool, whether to recursively convert any user functions or
  classes that the converted function may use.
* <b>`user_requested`</b>: bool, whether the conversion was explicitly requested by
  the user, as opposed to being performed as a result of other logic. This
  value always auto-resets resets to False in child conversions.
* <b>`optional_features`</b>: Union[Feature, Set[Feature]], controls the use of
  optional features in the conversion process. See Feature for available
  options.

<h2 id="__init__"><code>__init__</code></h2>

``` python
__init__(
    recursive=False,
    user_requested=False,
    internal_convert_user_code=True,
    optional_features=tf.autograph.experimental.Feature.ALL
)
```

Initialize self.  See help(type(self)) for accurate signature.




## Methods

<h3 id="__eq__"><code>__eq__</code></h3>

``` python
__eq__(other)
```

Return self==value.


<h3 id="as_tuple"><code>as_tuple</code></h3>

``` python
as_tuple()
```




<h3 id="call_options"><code>call_options</code></h3>

``` python
call_options()
```

Returns the corresponding options to be used for recursive conversion.


<h3 id="to_ast"><code>to_ast</code></h3>

``` python
to_ast()
```

Returns a representation of this object as an AST node.

The AST node encodes a constructor that would create an object with the
same contents.

#### Returns:

ast.Node


<h3 id="uses"><code>uses</code></h3>

``` python
uses(feature)
```






