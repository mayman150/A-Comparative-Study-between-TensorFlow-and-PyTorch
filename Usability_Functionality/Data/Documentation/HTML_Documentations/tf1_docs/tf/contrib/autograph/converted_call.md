<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.contrib.autograph.converted_call" />
<meta itemprop="path" content="Stable" />
</div>

# tf.contrib.autograph.converted_call

Compiles a function call inline.

``` python
tf.contrib.autograph.converted_call(
    f,
    options,
    args,
    kwargs,
    caller_fn_scope=None
)
```

<!-- Placeholder for "Used in" -->

For internal use only.

#### Args:


* <b>`f`</b>: The function to convert.
* <b>`options`</b>: converter.ConversionOptions
* <b>`args`</b>: Tuple, the original positional arguments of f
* <b>`kwargs`</b>: Dict, the original keyword arguments of f
* <b>`caller_fn_scope`</b>: Optional[function_wrappers.FunctionScope], the function
  scope of the converted function in which this call was originally made.


#### Returns:

Any, the result of executing a possibly-converted `f` with the given
  arguments.
