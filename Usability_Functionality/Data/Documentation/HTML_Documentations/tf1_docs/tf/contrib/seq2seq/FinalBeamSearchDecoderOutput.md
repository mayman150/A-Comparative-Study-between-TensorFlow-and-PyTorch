<div itemscope itemtype="http://developers.google.com/ReferenceObject">
<meta itemprop="name" content="tf.contrib.seq2seq.FinalBeamSearchDecoderOutput" />
<meta itemprop="path" content="Stable" />
<meta itemprop="property" content="predicted_ids"/>
<meta itemprop="property" content="beam_search_decoder_output"/>
</div>

# tf.contrib.seq2seq.FinalBeamSearchDecoderOutput

## Class `FinalBeamSearchDecoderOutput`

Final outputs returned by the beam search after all decoding is finished.



<!-- Placeholder for "Used in" -->


#### Args:


* <b>`predicted_ids`</b>: The final prediction. A tensor of shape
  `[batch_size, T, beam_width]` (or `[T, batch_size, beam_width]` if
  `output_time_major` is True). Beams are ordered from best to worst.
* <b>`beam_search_decoder_output`</b>: An instance of `BeamSearchDecoderOutput` that
  describes the state of the beam search.

## Properties

<h3 id="predicted_ids"><code>predicted_ids</code></h3>




<h3 id="beam_search_decoder_output"><code>beam_search_decoder_output</code></h3>






