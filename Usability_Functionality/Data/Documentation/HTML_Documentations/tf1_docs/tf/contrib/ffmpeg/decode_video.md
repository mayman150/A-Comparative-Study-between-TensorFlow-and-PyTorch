<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.contrib.ffmpeg.decode_video" />
<meta itemprop="path" content="Stable" />
</div>

# tf.contrib.ffmpeg.decode_video

Create an op that decodes the contents of a video file. (deprecated)

``` python
tf.contrib.ffmpeg.decode_video(contents)
```

<!-- Placeholder for "Used in" -->

Warning: THIS FUNCTION IS DEPRECATED. It will be removed after 2018-09-04.
Instructions for updating:
tf.contrib.ffmpeg will be removed in 2.0, the support for video and audio will continue to be provided in tensorflow-io: https://github.com/tensorflow/io

#### Args:


* <b>`contents`</b>: The binary contents of the video file to decode. This is a scalar.


#### Returns:

A rank-4 `Tensor` that has `[frames, height, width, 3]` RGB as output.
