<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.contrib.eager.ExecutionCallback" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="IGNORE"/>
<meta itemprop="property" content="PRINT"/>
<meta itemprop="property" content="RAISE"/>
<meta itemprop="property" content="WARN"/>
</div>

# tf.contrib.eager.ExecutionCallback

## Class `ExecutionCallback`

Valid callback actions.



<!-- Placeholder for "Used in" -->

These can be passed to `seterr` or `errstate` to create callbacks when
specific events occur (e.g. an operation produces `NaN`s).

IGNORE: take no action.
PRINT:  print a warning to `stdout`.
RAISE:  raise an error (e.g. `InfOrNanError`).
WARN:   print a warning using <a href="../../../tf/logging/warn.md"><code>tf.compat.v1.logging.warn</code></a>.

## Class Members

* `IGNORE` <a id="IGNORE"></a>
* `PRINT` <a id="PRINT"></a>
* `RAISE` <a id="RAISE"></a>
* `WARN` <a id="WARN"></a>
