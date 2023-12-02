<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.contrib.seq2seq.safe_cumprod" />
<meta itemprop="path" content="Stable" />
</div>

# tf.contrib.seq2seq.safe_cumprod

Computes cumprod of x in logspace using cumsum to avoid underflow.

``` python
tf.contrib.seq2seq.safe_cumprod(
    x,
    *args,
    **kwargs
)
```

<!-- Placeholder for "Used in" -->

The cumprod function and its gradient can result in numerical instabilities
when its argument has very small and/or zero values.  As long as the argument
is all positive, we can instead compute the cumulative product as
exp(cumsum(log(x))).  This function can be called identically to tf.cumprod.

#### Args:


* <b>`x`</b>: Tensor to take the cumulative product of.
* <b>`*args`</b>: Passed on to cumsum; these are identical to those in cumprod.
* <b>`**kwargs`</b>: Passed on to cumsum; these are identical to those in cumprod.


#### Returns:

Cumulative product of x.
