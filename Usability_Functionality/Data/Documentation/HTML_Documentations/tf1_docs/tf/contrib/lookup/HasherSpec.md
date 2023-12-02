<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.contrib.lookup.HasherSpec" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="hasher"/>
<meta itemprop="property" content="key"/>
</div>

# tf.contrib.lookup.HasherSpec

## Class `HasherSpec`

A structure for the spec of the hashing function to use for hash buckets.



<!-- Placeholder for "Used in" -->

`hasher` is the name of the hashing function to use (eg. "fasthash",
"stronghash").
`key` is optional and specify the key to use for the hash function if
supported, currently only used by a strong hash.

#### Fields:


* <b>`hasher`</b>: The hasher name to use.
* <b>`key`</b>: The key to be used by the hashing function, if required.

## Properties

<h3 id="hasher"><code>hasher</code></h3>




<h3 id="key"><code>key</code></h3>






