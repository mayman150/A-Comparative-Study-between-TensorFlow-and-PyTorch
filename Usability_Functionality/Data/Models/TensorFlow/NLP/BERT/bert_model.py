from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import numpy as np
import tensorflow as tf

from ...activations_tf import get_tf_activation
from ...modeling_tf_outputs import (
    TFBaseModelOutputWithPastAndCrossAttentions,
    TFBaseModelOutputWithPoolingAndCrossAttentions,
    TFCausalLMOutputWithCrossAttentions,
    TFMaskedLMOutput,
    TFMultipleChoiceModelOutput,
    TFNextSentencePredictorOutput,
    TFQuestionAnsweringModelOutput,
    TFSequenceClassifierOutput,
    TFTokenClassifierOutput,
)
from ...modeling_tf_utils import (
    TFCausalLanguageModelingLoss,
    TFMaskedLanguageModelingLoss,
    TFModelInputType,
    TFMultipleChoiceLoss,
    TFNextSentencePredictionLoss,
    TFPreTrainedModel,
    TFQuestionAnsweringLoss,
    TFSequenceClassificationLoss,
    TFTokenClassificationLoss,
    get_initializer,
    keras_serializable,
    unpack_inputs,
)
from ...tf_utils import check_embeddings_within_bounds, shape_list, stable_softmax
from ...utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from .configuration_bert import BertConfig


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "bert-base-uncased"
_CONFIG_FOR_DOC = "BertConfig"

# TokenClassification docstring
_CHECKPOINT_FOR_TOKEN_CLASSIFICATION = "dbmdz/bert-large-cased-finetuned-conll03-english"
_TOKEN_CLASS_EXPECTED_OUTPUT = (
    "['O', 'I-ORG', 'I-ORG', 'I-ORG', 'O', 'O', 'O', 'O', 'O', 'I-LOC', 'O', 'I-LOC', 'I-LOC'] "
)
_TOKEN_CLASS_EXPECTED_LOSS = 0.01

# QuestionAnswering docstring
_CHECKPOINT_FOR_QA = "ydshieh/bert-base-cased-squad2"
_QA_EXPECTED_OUTPUT = "'a nice puppet'"
_QA_EXPECTED_LOSS = 7.41
_QA_TARGET_START_INDEX = 14
_QA_TARGET_END_INDEX = 15

# SequenceClassification docstring
_CHECKPOINT_FOR_SEQUENCE_CLASSIFICATION = "ydshieh/bert-base-uncased-yelp-polarity"
_SEQ_CLASS_EXPECTED_OUTPUT = "'LABEL_1'"
_SEQ_CLASS_EXPECTED_LOSS = 0.01

TF_BERT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "bert-base-uncased",
    "bert-large-uncased",
    "bert-base-cased",
    "bert-large-cased",
    "bert-base-multilingual-uncased",
    "bert-base-multilingual-cased",
    "bert-base-chinese",
    "bert-base-german-cased",
    "bert-large-uncased-whole-word-masking",
    "bert-large-cased-whole-word-masking",
    "bert-large-uncased-whole-word-masking-finetuned-squad",
    "bert-large-cased-whole-word-masking-finetuned-squad",
    "bert-base-cased-finetuned-mrpc",
    "cl-tohoku/bert-base-japanese",
    "cl-tohoku/bert-base-japanese-whole-word-masking",
    "cl-tohoku/bert-base-japanese-char",
    "cl-tohoku/bert-base-japanese-char-whole-word-masking",
    "TurkuNLP/bert-base-finnish-cased-v1",
    "TurkuNLP/bert-base-finnish-uncased-v1",
    "wietsedv/bert-base-dutch-cased",
    # See all BERT models at https://huggingface.co/models?filter=bert
]

#13
class TFBertPreTrainingLoss:
    """
    Loss function suitable for BERT-like pretraining, that is, the task of pretraining a language model by combining
    NSP + MLM. .. note:: Any label of -100 will be ignored (along with the corresponding logits) in the loss
    computation.
    """
    #13
    def hf_compute_loss(self, labels: tf.Tensor, logits: tf.Tensor) -> tf.Tensor:
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction=tf.keras.losses.Reduction.NONE
        )

        # Clip negative labels to zero here to avoid NaNs and errors - those positions will get masked later anyway
        unmasked_lm_losses = loss_fn(y_true=tf.nn.relu(labels["labels"]), y_pred=logits[0])
        # make sure only labels that are not equal to -100
        # are taken into account for the loss computation
        lm_loss_mask = tf.cast(labels["labels"] != -100, dtype=unmasked_lm_losses.dtype)
        masked_lm_losses = unmasked_lm_losses * lm_loss_mask
        reduced_masked_lm_loss = tf.reduce_sum(masked_lm_losses) / tf.reduce_sum(lm_loss_mask)

        # Clip negative labels to zero here to avoid NaNs and errors - those positions will get masked later anyway
        unmasked_ns_loss = loss_fn(y_true=tf.nn.relu(labels["next_sentence_label"]), y_pred=logits[1])
        ns_loss_mask = tf.cast(labels["next_sentence_label"] != -100, dtype=unmasked_ns_loss.dtype)
        masked_ns_loss = unmasked_ns_loss * ns_loss_mask

        reduced_masked_ns_loss = tf.reduce_sum(masked_ns_loss) / tf.reduce_sum(ns_loss_mask)

        return tf.reshape(reduced_masked_lm_loss + reduced_masked_ns_loss, (1,))

#19
class TFBertEmbeddings(tf.keras.layers.Layer):
    """Construct the embeddings from word, position and token_type embeddings."""
    #6 first two functions
    def __init__(self, config: BertConfig, **kwargs):
        super().__init__(**kwargs)

        self.config = config
        self.hidden_size = config.hidden_size
        self.max_position_embeddings = config.max_position_embeddings
        self.initializer_range = config.initializer_range
        self.LayerNorm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")
        self.dropout = tf.keras.layers.Dropout(rate=config.hidden_dropout_prob)
    
    def build(self, input_shape: tf.TensorShape):
        with tf.name_scope("word_embeddings"):
            self.weight = self.add_weight(
                name="weight",
                shape=[self.config.vocab_size, self.hidden_size],
                initializer=get_initializer(self.initializer_range),
            )

        with tf.name_scope("token_type_embeddings"):
            self.token_type_embeddings = self.add_weight(
                name="embeddings",
                shape=[self.config.type_vocab_size, self.hidden_size],
                initializer=get_initializer(self.initializer_range),
            )

        with tf.name_scope("position_embeddings"):
            self.position_embeddings = self.add_weight(
                name="embeddings",
                shape=[self.max_position_embeddings, self.hidden_size],
                initializer=get_initializer(self.initializer_range),
            )

        super().build(input_shape)
    #13
    def call(
        self,
        input_ids: tf.Tensor = None,
        position_ids: tf.Tensor = None,
        token_type_ids: tf.Tensor = None,
        inputs_embeds: tf.Tensor = None,
        past_key_values_length=0,
        training: bool = False,
    ) -> tf.Tensor:
        """
        Applies embedding based on inputs tensor.

        Returns:
            final_embeddings (`tf.Tensor`): output embedding tensor.
        """
        if input_ids is None and inputs_embeds is None:
            raise ValueError("Need to provide either `input_ids` or `input_embeds`.")

        if input_ids is not None:
            check_embeddings_within_bounds(input_ids, self.config.vocab_size)
            inputs_embeds = tf.gather(params=self.weight, indices=input_ids)

        input_shape = shape_list(inputs_embeds)[:-1]

        if token_type_ids is None:
            token_type_ids = tf.fill(dims=input_shape, value=0)

        if position_ids is None:
            position_ids = tf.expand_dims(
                tf.range(start=past_key_values_length, limit=input_shape[1] + past_key_values_length), axis=0
            )

        position_embeds = tf.gather(params=self.position_embeddings, indices=position_ids)
        token_type_embeds = tf.gather(params=self.token_type_embeddings, indices=token_type_ids)
        final_embeddings = inputs_embeds + position_embeds + token_type_embeds
        final_embeddings = self.LayerNorm(inputs=final_embeddings)
        final_embeddings = self.dropout(inputs=final_embeddings, training=training)

        return final_embeddings

#31
class TFBertSelfAttention(tf.keras.layers.Layer):
    #4
    def __init__(self, config: BertConfig, **kwargs):
        super().__init__(**kwargs)

        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number "
                f"of attention heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.sqrt_att_head_size = math.sqrt(self.attention_head_size)

        self.query = tf.keras.layers.Dense(#1
            units=self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name="query"
        )
        self.key = tf.keras.layers.Dense(#1
            units=self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name="key"
        )
        self.value = tf.keras.layers.Dense(#1
            units=self.all_head_size, kernel_initializer=get_initializer(config.initializer_range), name="value"
        )
        self.dropout = tf.keras.layers.Dropout(rate=config.attention_probs_dropout_prob)#1

        self.is_decoder = config.is_decoder
    #2
    def transpose_for_scores(self, tensor: tf.Tensor, batch_size: int) -> tf.Tensor:
        # Reshape from [batch_size, seq_length, all_head_size] to [batch_size, seq_length, num_attention_heads, attention_head_size]
        tensor = tf.reshape(tensor=tensor, shape=(batch_size, -1, self.num_attention_heads, self.attention_head_size))

        # Transpose the tensor from [batch_size, seq_length, num_attention_heads, attention_head_size] to [batch_size, num_attention_heads, seq_length, attention_head_size]
        return tf.transpose(tensor, perm=[0, 2, 1, 3])
    #27
    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: tf.Tensor,
        head_mask: tf.Tensor,
        encoder_hidden_states: tf.Tensor,
        encoder_attention_mask: tf.Tensor,
        past_key_value: Tuple[tf.Tensor],
        output_attentions: bool,
        training: bool = False,
    ) -> Tuple[tf.Tensor]:
        batch_size = shape_list(hidden_states)[0]
        mixed_query_layer = self.query(inputs=hidden_states)#1

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(inputs=encoder_hidden_states), batch_size)#2
            value_layer = self.transpose_for_scores(self.value(inputs=encoder_hidden_states), batch_size)#2
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(inputs=hidden_states), batch_size)#2
            value_layer = self.transpose_for_scores(self.value(inputs=hidden_states), batch_size)#2
            key_layer = tf.concat([past_key_value[0], key_layer], axis=2)#1
            value_layer = tf.concat([past_key_value[1], value_layer], axis=2)#1
        else:
            key_layer = self.transpose_for_scores(self.key(inputs=hidden_states), batch_size)#2
            value_layer = self.transpose_for_scores(self.value(inputs=hidden_states), batch_size)#2

        query_layer = self.transpose_for_scores(mixed_query_layer, batch_size)#2

        if self.is_decoder:
            past_key_value = (key_layer, value_layer)

        attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)#1
        dk = tf.cast(self.sqrt_att_head_size, dtype=attention_scores.dtype)#1
        attention_scores = tf.divide(attention_scores, dk)#1

        if attention_mask is not None:
            attention_scores = tf.add(attention_scores, attention_mask)#1
        attention_probs = stable_softmax(logits=attention_scores, axis=-1)#1
        attention_probs = self.dropout(inputs=attention_probs, training=training)#1
        if head_mask is not None:
            attention_probs = tf.multiply(attention_probs, head_mask)#1

        attention_output = tf.matmul(attention_probs, value_layer)#1
        attention_output = tf.transpose(attention_output, perm=[0, 2, 1, 3])#1

        # (batch_size, seq_len_q, all_head_size)
        attention_output = tf.reshape(tensor=attention_output, shape=(batch_size, -1, self.all_head_size))#1
        outputs = (attention_output, attention_probs) if output_attentions else (attention_output,)

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs

#6
class TFBertSelfOutput(tf.keras.layers.Layer):
    def __init__(self, config: BertConfig, **kwargs):#3
        super().__init__(**kwargs)

        self.dense = tf.keras.layers.Dense(
            units=config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )
        self.LayerNorm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")
        self.dropout = tf.keras.layers.Dropout(rate=config.hidden_dropout_prob)
    #3
    def call(self, hidden_states: tf.Tensor, input_tensor: tf.Tensor, training: bool = False) -> tf.Tensor:
        hidden_states = self.dense(inputs=hidden_states)
        hidden_states = self.dropout(inputs=hidden_states, training=training)
        hidden_states = self.LayerNorm(inputs=hidden_states + input_tensor)

        return hidden_states

#31*3 = 93
class TFBertAttention(tf.keras.layers.Layer):
    def __init__(self, config: BertConfig, **kwargs):
        super().__init__(**kwargs)

        self.self_attention = TFBertSelfAttention(config, name="self")#31
        self.dense_output = TFBertSelfOutput(config, name="output")#31

    def prune_heads(self, heads):
        raise NotImplementedError

    def call(
        self,
        input_tensor: tf.Tensor,
        attention_mask: tf.Tensor,
        head_mask: tf.Tensor,
        encoder_hidden_states: tf.Tensor,
        encoder_attention_mask: tf.Tensor,
        past_key_value: Tuple[tf.Tensor],
        output_attentions: bool,
        training: bool = False,
    ) -> Tuple[tf.Tensor]:
        self_outputs = self.self_attention(
            hidden_states=input_tensor,
            attention_mask=attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            training=training,
        )
        attention_output = self.dense_output(#31
            hidden_states=self_outputs[0], input_tensor=input_tensor, training=training
        )
        # add attentions (possibly with past_key_value) if we output them
        outputs = (attention_output,) + self_outputs[1:]

        return outputs

#3
class TFBertIntermediate(tf.keras.layers.Layer):
    def __init__(self, config: BertConfig, **kwargs):
        super().__init__(**kwargs)

        self.dense = tf.keras.layers.Dense(
            units=config.intermediate_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )#1

        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = get_tf_activation(config.hidden_act)#1
        else:
            self.intermediate_act_fn = config.hidden_act

    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        hidden_states = self.dense(inputs=hidden_states)#1
        hidden_states = self.intermediate_act_fn(hidden_states)#1

        return hidden_states

#5
class TFBertOutput(tf.keras.layers.Layer):
    def __init__(self, config: BertConfig, **kwargs):
        super().__init__(**kwargs)

        self.dense = tf.keras.layers.Dense(#1
            units=config.hidden_size, kernel_initializer=get_initializer(config.initializer_range), name="dense"
        )
        self.LayerNorm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")#1
        self.dropout = tf.keras.layers.Dropout(rate=config.hidden_dropout_prob)#1

    def call(self, hidden_states: tf.Tensor, input_tensor: tf.Tensor, training: bool = False) -> tf.Tensor:
        hidden_states = self.dense(inputs=hidden_states)#1
        hidden_states = self.dropout(inputs=hidden_states, training=training)#1
        hidden_states = self.LayerNorm(inputs=hidden_states + input_tensor)#1

        return hidden_states

#189+194 = 383
class TFBertLayer(tf.keras.layers.Layer):
    def __init__(self, config: BertConfig, **kwargs):#93*2 + 3 + 5 = 194
        super().__init__(**kwargs)

        self.attention = TFBertAttention(config, name="attention")#93
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            if not self.is_decoder:
                raise ValueError(f"{self} should be used as a decoder model if cross attention is added")
            self.crossattention = TFBertAttention(config, name="crossattention")#93
        self.intermediate = TFBertIntermediate(config, name="intermediate")#3
        self.bert_output = TFBertOutput(config, name="output")#5

    def call(#96+93 = 189
        self,
        hidden_states: tf.Tensor,
        attention_mask: tf.Tensor,
        head_mask: tf.Tensor,
        encoder_hidden_states: tf.Tensor | None,
        encoder_attention_mask: tf.Tensor | None,
        past_key_value: Tuple[tf.Tensor] | None,
        output_attentions: bool,
        training: bool = False,
    ) -> Tuple[tf.Tensor]:
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        self_attention_outputs = self.attention(#93
            input_tensor=hidden_states,
            attention_mask=attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past_key_value=self_attn_past_key_value,
            output_attentions=output_attentions,
            training=training,
        )
        attention_output = self_attention_outputs[0]

        # if decoder, the last output is tuple of self-attn cache
        if self.is_decoder:
            outputs = self_attention_outputs[1:-1]
            present_key_value = self_attention_outputs[-1]
        else:
            outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        cross_attn_present_key_value = None
        if self.is_decoder and encoder_hidden_states is not None:
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers"
                    " by setting `config.add_cross_attention=True`"
                )

            # cross_attn cached key/values tuple is at positions 3,4 of past_key_value tuple
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            cross_attention_outputs = self.crossattention(#93
                input_tensor=attention_output,
                attention_mask=attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                past_key_value=cross_attn_past_key_value,
                output_attentions=output_attentions,
                training=training,
            )
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:-1]  # add cross attentions if we output attention weights

            # add cross-attn cache to positions 3,4 of present_key_value tuple
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value

        intermediate_output = self.intermediate(hidden_states=attention_output)#3
        layer_output = self.bert_output(
            hidden_states=intermediate_output, input_tensor=attention_output, training=training
        )
        outputs = (layer_output,) + outputs  # add attentions if we output them

        # if decoder, return the attn key/values as the last output
        if self.is_decoder:
            outputs = outputs + (present_key_value,)

        return outputs

#384
class TFBertEncoder(tf.keras.layers.Layer):
    def __init__(self, config: BertConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.layer = [TFBertLayer(config, name=f"layer_._{i}") for i in range(config.num_hidden_layers)]#383

    def call(
        self,
        hidden_states: tf.Tensor,
        attention_mask: tf.Tensor,
        head_mask: tf.Tensor,
        encoder_hidden_states: tf.Tensor | None,
        encoder_attention_mask: tf.Tensor | None,
        past_key_values: Tuple[Tuple[tf.Tensor]] | None,
        use_cache: Optional[bool],
        output_attentions: bool,
        output_hidden_states: bool,
        return_dict: bool,
        training: bool = False,
    ) -> Union[TFBaseModelOutputWithPastAndCrossAttentions, Tuple[tf.Tensor]]:
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        next_decoder_cache = () if use_cache else None
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            past_key_value = past_key_values[i] if past_key_values is not None else None

            layer_outputs = layer_module(#1
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                head_mask=head_mask[i],
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                training=training,
            )
            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention and encoder_hidden_states is not None:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v for v in [hidden_states, all_hidden_states, all_attentions, all_cross_attentions] if v is not None
            )

        return TFBaseModelOutputWithPastAndCrossAttentions(#1
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            cross_attentions=all_cross_attentions,
        )

#2
class TFBertPooler(tf.keras.layers.Layer):
    def __init__(self, config: BertConfig, **kwargs):
        super().__init__(**kwargs)

        self.dense = tf.keras.layers.Dense(#1
            units=config.hidden_size,
            kernel_initializer=get_initializer(config.initializer_range),
            activation="tanh",
            name="dense",
        )

    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(inputs=first_token_tensor)#1

        return pooled_output

#5
class TFBertPredictionHeadTransform(tf.keras.layers.Layer):
    def __init__(self, config: BertConfig, **kwargs):
        super().__init__(**kwargs)

        self.dense = tf.keras.layers.Dense(
            units=config.hidden_size,
            kernel_initializer=get_initializer(config.initializer_range),
            name="dense",
        )

        if isinstance(config.hidden_act, str):
            self.transform_act_fn = get_tf_activation(config.hidden_act)
        else:
            self.transform_act_fn = config.hidden_act

        self.LayerNorm = tf.keras.layers.LayerNormalization(epsilon=config.layer_norm_eps, name="LayerNorm")

    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        hidden_states = self.dense(inputs=hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(inputs=hidden_states)

        return hidden_states

#17
class TFBertLMPredictionHead(tf.keras.layers.Layer):
    def __init__(self, config: BertConfig, input_embeddings: tf.keras.layers.Layer, **kwargs):
        super().__init__(**kwargs)

        self.config = config
        self.hidden_size = config.hidden_size

        self.transform = TFBertPredictionHeadTransform(config, name="transform")#5

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.input_embeddings = input_embeddings

    def build(self, input_shape: tf.TensorShape):
        self.bias = self.add_weight(shape=(self.config.vocab_size,), initializer="zeros", trainable=True, name="bias")

        super().build(input_shape)

    def get_output_embeddings(self) -> tf.keras.layers.Layer:
        return self.input_embeddings

    def set_output_embeddings(self, value: tf.Variable):
        self.input_embeddings.weight = value
        self.input_embeddings.vocab_size = shape_list(value)[0]

    def get_bias(self) -> Dict[str, tf.Variable]:
        return {"bias": self.bias}

    def set_bias(self, value: tf.Variable):
        self.bias = value["bias"]
        self.config.vocab_size = shape_list(value["bias"])[0]

    def call(self, hidden_states: tf.Tensor) -> tf.Tensor:
        hidden_states = self.transform(hidden_states=hidden_states)
        seq_length = shape_list(hidden_states)[1]
        hidden_states = tf.reshape(tensor=hidden_states, shape=[-1, self.hidden_size])
        hidden_states = tf.matmul(a=hidden_states, b=self.input_embeddings.weight, transpose_b=True)
        hidden_states = tf.reshape(tensor=hidden_states, shape=[-1, seq_length, self.config.vocab_size])
        hidden_states = tf.nn.bias_add(value=hidden_states, bias=self.bias)

        return hidden_states

#17*2 = 34
class TFBertMLMHead(tf.keras.layers.Layer):
    def __init__(self, config: BertConfig, input_embeddings: tf.keras.layers.Layer, **kwargs):
        super().__init__(**kwargs)

        self.predictions = TFBertLMPredictionHead(config, input_embeddings, name="predictions")#17

    def call(self, sequence_output: tf.Tensor) -> tf.Tensor:
        prediction_scores = self.predictions(hidden_states=sequence_output)#17

        return prediction_scores

#3
class TFBertNSPHead(tf.keras.layers.Layer):
    def __init__(self, config: BertConfig, **kwargs):
        super().__init__(**kwargs)

        self.seq_relationship = tf.keras.layers.Dense(
            units=2,
            kernel_initializer=get_initializer(config.initializer_range),
            name="seq_relationship",
        )

    def call(self, pooled_output: tf.Tensor) -> tf.Tensor:
        seq_relationship_score = self.seq_relationship(inputs=pooled_output)

        return seq_relationship_score

#418+19+19+384+2 = 842
@keras_serializable
class TFBertMainLayer(tf.keras.layers.Layer):
    config_class = BertConfig

    def __init__(self, config: BertConfig, add_pooling_layer: bool = True, **kwargs):
        super().__init__(**kwargs)

        self.config = config
        self.is_decoder = config.is_decoder

        self.embeddings = TFBertEmbeddings(config, name="embeddings")#19
        self.encoder = TFBertEncoder(config, name="encoder")#384
        self.pooler = TFBertPooler(config, name="pooler") if add_pooling_layer else None#2

    def get_input_embeddings(self) -> tf.keras.layers.Layer:#19
        return self.embeddings

    def set_input_embeddings(self, value: tf.Variable):#19
        self.embeddings.weight = value
        self.embeddings.vocab_size = shape_list(value)[0]

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        raise NotImplementedError

    @unpack_inputs
    def call(
        self,
        input_ids: TFModelInputType | None = None,
        attention_mask: np.ndarray | tf.Tensor | None = None,
        token_type_ids: np.ndarray | tf.Tensor | None = None,
        position_ids: np.ndarray | tf.Tensor | None = None,
        head_mask: np.ndarray | tf.Tensor | None = None,
        inputs_embeds: np.ndarray | tf.Tensor | None = None,
        encoder_hidden_states: np.ndarray | tf.Tensor | None = None,
        encoder_attention_mask: np.ndarray | tf.Tensor | None = None,
        past_key_values: Optional[Tuple[Tuple[Union[np.ndarray, tf.Tensor]]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        training: bool = False,
    ) -> Union[TFBaseModelOutputWithPoolingAndCrossAttentions, Tuple[tf.Tensor]]:
        if not self.config.is_decoder:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = shape_list(input_ids)
        elif inputs_embeds is not None:
            input_shape = shape_list(inputs_embeds)[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape

        if past_key_values is None:
            past_key_values_length = 0
            past_key_values = [None] * len(self.encoder.layer)
        else:
            past_key_values_length = shape_list(past_key_values[0][0])[-2]

        if attention_mask is None:
            attention_mask = tf.fill(dims=(batch_size, seq_length + past_key_values_length), value=1)#1

        if token_type_ids is None:
            token_type_ids = tf.fill(dims=input_shape, value=0)#1

        embedding_output = self.embeddings(#19
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
            training=training,
        )

        attention_mask_shape = shape_list(attention_mask)

        mask_seq_length = seq_length + past_key_values_length
        if self.is_decoder:
            seq_ids = tf.range(mask_seq_length)#1
            causal_mask = tf.less_equal(#1
                tf.tile(seq_ids[None, None, :], (batch_size, mask_seq_length, 1)),
                seq_ids[None, :, None],
            )
            causal_mask = tf.cast(causal_mask, dtype=attention_mask.dtype)#1
            extended_attention_mask = causal_mask * attention_mask[:, None, :]
            attention_mask_shape = shape_list(extended_attention_mask)
            extended_attention_mask = tf.reshape(#1
                extended_attention_mask, (attention_mask_shape[0], 1, attention_mask_shape[1], attention_mask_shape[2])
            )
            if past_key_values[0] is not None:
                # attention_mask needs to be sliced to the shape `[batch_size, 1, from_seq_length - cached_seq_length, to_seq_length]
                extended_attention_mask = extended_attention_mask[:, :, -seq_length:, :]
        else:
            extended_attention_mask = tf.reshape(#1
                attention_mask, (attention_mask_shape[0], 1, 1, attention_mask_shape[1])
            )
        extended_attention_mask = tf.cast(extended_attention_mask, dtype=embedding_output.dtype)#1
        one_cst = tf.constant(1.0, dtype=embedding_output.dtype)#1
        ten_thousand_cst = tf.constant(-10000.0, dtype=embedding_output.dtype)#1
        extended_attention_mask = tf.multiply(tf.subtract(one_cst, extended_attention_mask), ten_thousand_cst)#1
        if self.is_decoder and encoder_attention_mask is not None:

            encoder_attention_mask = tf.cast(encoder_attention_mask, dtype=extended_attention_mask.dtype)#1
            num_dims_encoder_attention_mask = len(shape_list(encoder_attention_mask))
            if num_dims_encoder_attention_mask == 3:
                encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
            if num_dims_encoder_attention_mask == 2:
                encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]
            encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * -10000.0
        else:
            encoder_extended_attention_mask = None
        if head_mask is not None:
            raise NotImplementedError
        else:
            head_mask = [None] * self.config.num_hidden_layers

        encoder_outputs = self.encoder(#384 + 31 = 415
            hidden_states=embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            training=training,
        )

        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(hidden_states=sequence_output) if self.pooler is not None else None#2

        if not return_dict:
            return (
                sequence_output,
                pooled_output,
            ) + encoder_outputs[1:]

        return TFBaseModelOutputWithPoolingAndCrossAttentions(#418
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )