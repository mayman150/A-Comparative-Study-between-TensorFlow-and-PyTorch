description: Registry of :class:~absl.flags.Flag objects.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.compat.v1.flags.FLAGS" />
<meta itemprop="path" content="Stable" />
</div>

# tf.compat.v1.flags.FLAGS

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">

</table>



Registry of :class:`~absl.flags.Flag` objects.


<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.compat.v1.flags.FLAGS(
    *args, **kwargs
)
</code></pre>



<!-- Placeholder for "Used in" -->

A :class:`FlagValues` can then scan command line arguments, passing flag
arguments through to the 'Flag' objects that it owns.  It also
provides easy access to the flag values.  Typically only one
:class:`FlagValues` object is needed by an application:
:const:`FLAGS`.

This class is heavily overloaded:

:class:`Flag` objects are registered via ``__setitem__``::

     FLAGS['longname'] = x   # register a new flag

The ``.value`` attribute of the registered :class:`~absl.flags.Flag` objects
can be accessed as attributes of this :class:`FlagValues` object, through
``__getattr__``.  Both the long and short name of the original
:class:`~absl.flags.Flag` objects can be used to access its value::

     FLAGS.longname  # parsed flag value
     FLAGS.x  # parsed flag value (short name)

Command line arguments are scanned and passed to the registered
:class:`~absl.flags.Flag` objects through the ``__call__`` method.  Unparsed
arguments, including ``argv[0]`` (e.g. the program name) are returned::

     argv = FLAGS(sys.argv)  # scan command line arguments

The original registered :class:`~absl.flags.Flag` objects can be retrieved
through the use of the dictionary-like operator, ``__getitem__``::

     x = FLAGS['longname']   # access the registered Flag object

The ``str()`` operator of a :class:`absl.flags.FlagValues` object provides
help for all of the registered :class:`~absl.flags.Flag` objects.