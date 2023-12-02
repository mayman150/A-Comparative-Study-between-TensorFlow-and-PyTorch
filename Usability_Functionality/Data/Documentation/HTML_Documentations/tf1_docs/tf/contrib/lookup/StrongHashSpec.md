<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.contrib.lookup.StrongHashSpec" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="hasher"/>
<meta itemprop="property" content="key"/>
</div>

# tf.contrib.lookup.StrongHashSpec

## Class `StrongHashSpec`

A structure to specify a key of the strong keyed hash spec.

Inherits From: [`HasherSpec`](../../../tf/contrib/lookup/HasherSpec.md)

<!-- Placeholder for "Used in" -->

The strong hash requires a `key`, which is a list of 2 unsigned integer
numbers. These should be non-zero; random numbers generated from random.org
would be a fine choice.

#### Fields:


* <b>`key`</b>: The key to be used by the keyed hashing function.

## Properties

<h3 id="hasher"><code>hasher</code></h3>




<h3 id="key"><code>key</code></h3>






