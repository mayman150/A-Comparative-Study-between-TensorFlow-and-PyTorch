description: Downloads a file from a URL if it not already in the cache.

<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.keras.utils.get_file" />
<meta itemprop="path" content="Stable" />
</div>

# tf.keras.utils.get_file

<!-- Insert buttons and diff -->

<table class="tfo-notebook-buttons tfo-api nocontent" align="left">
<td>
  <a target="_blank" href="https://github.com/keras-team/keras/tree/v2.15.0/keras/utils/data_utils.py#L193-L377">
    <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
    View source on GitHub
  </a>
</td>
</table>



Downloads a file from a URL if it not already in the cache.


<pre class="devsite-click-to-copy prettyprint lang-py tfo-signature-link">
<code>tf.keras.utils.get_file(
    fname=None,
    origin=None,
    untar=False,
    md5_hash=None,
    file_hash=None,
    cache_subdir=&#x27;datasets&#x27;,
    hash_algorithm=&#x27;auto&#x27;,
    extract=False,
    archive_format=&#x27;auto&#x27;,
    cache_dir=None
)
</code></pre>



<!-- Placeholder for "Used in" -->

By default the file at the url `origin` is downloaded to the
cache_dir `~/.keras`, placed in the cache_subdir `datasets`,
and given the filename `fname`. The final location of a file
`example.txt` would therefore be `~/.keras/datasets/example.txt`.

Files in tar, tar.gz, tar.bz, and zip formats can also be extracted.
Passing a hash will verify the file after download. The command line
programs `shasum` and `sha256sum` can compute the hash.

#### Example:



```python
path_to_downloaded_file = tf.keras.utils.get_file(
    origin="https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz",
    extract=True,
)
```

<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Args</h2></th></tr>

<tr>
<td>
`fname`<a id="fname"></a>
</td>
<td>
Name of the file. If an absolute path `/path/to/file.txt` is
specified the file will be saved at that location. If `None`, the
name of the file at `origin` will be used.
</td>
</tr><tr>
<td>
`origin`<a id="origin"></a>
</td>
<td>
Original URL of the file.
</td>
</tr><tr>
<td>
`untar`<a id="untar"></a>
</td>
<td>
Deprecated in favor of `extract` argument.
boolean, whether the file should be decompressed
</td>
</tr><tr>
<td>
`md5_hash`<a id="md5_hash"></a>
</td>
<td>
Deprecated in favor of `file_hash` argument.
md5 hash of the file for verification
</td>
</tr><tr>
<td>
`file_hash`<a id="file_hash"></a>
</td>
<td>
The expected hash string of the file after download.
The sha256 and md5 hash algorithms are both supported.
</td>
</tr><tr>
<td>
`cache_subdir`<a id="cache_subdir"></a>
</td>
<td>
Subdirectory under the Keras cache dir where the file is
saved. If an absolute path `/path/to/folder` is
specified the file will be saved at that location.
</td>
</tr><tr>
<td>
`hash_algorithm`<a id="hash_algorithm"></a>
</td>
<td>
Select the hash algorithm to verify the file.
options are `'md5'`, `'sha256'`, and `'auto'`.
The default 'auto' detects the hash algorithm in use.
</td>
</tr><tr>
<td>
`extract`<a id="extract"></a>
</td>
<td>
True tries extracting the file as an Archive, like tar or zip.
</td>
</tr><tr>
<td>
`archive_format`<a id="archive_format"></a>
</td>
<td>
Archive format to try for extracting the file.
Options are `'auto'`, `'tar'`, `'zip'`, and `None`.
`'tar'` includes tar, tar.gz, and tar.bz files.
The default `'auto'` corresponds to `['tar', 'zip']`.
None or an empty list will return no matches found.
</td>
</tr><tr>
<td>
`cache_dir`<a id="cache_dir"></a>
</td>
<td>
Location to store cached files, when None it
defaults to `~/.keras/`.
</td>
</tr>
</table>



<!-- Tabular view -->
 <table class="responsive fixed orange">
<colgroup><col width="214px"><col></colgroup>
<tr><th colspan="2"><h2 class="add-link">Returns</h2></th></tr>
<tr class="alt">
<td colspan="2">
Path to the downloaded file.
</td>
</tr>

</table>


⚠️ **Warning on malicious downloads** ⚠️

Downloading something from the Internet carries a risk.
NEVER download a file/archive if you do not trust the source.
We recommend that you specify the `file_hash` argument
(if the hash of the source file is known) to make sure that the file you
are getting is the one you expect.