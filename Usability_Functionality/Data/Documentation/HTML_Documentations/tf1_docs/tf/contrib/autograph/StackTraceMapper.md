<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.contrib.autograph.StackTraceMapper" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="__enter__"/>
<meta itemprop="property" content="__exit__"/>
<meta itemprop="property" content="__init__"/>
<meta itemprop="property" content="get_effective_source_map"/>
<meta itemprop="property" content="reset"/>
</div>

# tf.contrib.autograph.StackTraceMapper

## Class `StackTraceMapper`

Remaps generated code to code it originated from.



<!-- Placeholder for "Used in" -->


<h2 id="__init__"><code>__init__</code></h2>

``` python
__init__(converted_fn)
```

Initialize self.  See help(type(self)) for accurate signature.




## Methods

<h3 id="__enter__"><code>__enter__</code></h3>

``` python
__enter__()
```




<h3 id="__exit__"><code>__exit__</code></h3>

``` python
__exit__(
    unused_type,
    unused_value,
    unused_traceback
)
```




<h3 id="get_effective_source_map"><code>get_effective_source_map</code></h3>

``` python
get_effective_source_map()
```

Returns a map (filename, lineno) -> (filename, lineno, function_name).


<h3 id="reset"><code>reset</code></h3>

``` python
reset()
```






