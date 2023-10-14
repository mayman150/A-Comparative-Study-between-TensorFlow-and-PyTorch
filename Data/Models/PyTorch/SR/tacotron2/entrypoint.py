# *****************************************************************************
#  Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are met:
#      * Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#      * Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#      * Neither the name of the NVIDIA CORPORATION nor the
#        names of its contributors may be used to endorse or promote products
#        derived from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
#  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
#  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#  DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
#  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
#  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# *****************************************************************************

import urllib.request
import torch
import os
import sys

#from https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/SpeechSynthesis/Tacotron2/inference.py
def checkpoint_from_distributed(state_dict):
    """
    Checks whether checkpoint was generated by DistributedDataParallel. DDP
    wraps model in additional "module.", it needs to be unwrapped for single
    GPU inference.
    :param state_dict: model's state dict
    """
    ret = False
    for key, _ in state_dict.items():
        if key.find('module.') != -1:
            ret = True
            break
    return ret


# from https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/SpeechSynthesis/Tacotron2/inference.py
def unwrap_distributed(state_dict):
    """
    Unwraps model from DistributedDataParallel.
    DDP wraps model in additional "module.", it needs to be removed for single
    GPU inference.
    :param state_dict: model's state dict
    """
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace('module.1.', '')
        new_key = new_key.replace('module.', '')
        new_state_dict[new_key] = value
    return new_state_dict

def _download_checkpoint(checkpoint, force_reload):
    model_dir = os.path.join(torch.hub._get_torch_home(), 'checkpoints')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    ckpt_file = os.path.join(model_dir, os.path.basename(checkpoint))
    if not os.path.exists(ckpt_file) or force_reload:
        sys.stderr.write('Downloading checkpoint from {}\n'.format(checkpoint))
        urllib.request.urlretrieve(checkpoint, ckpt_file)
    return ckpt_file

def nvidia_tacotron2(pretrained=True, **kwargs):
    """Constructs a Tacotron 2 model (nn.module with additional infer(input) method).
    For detailed information on model input and output, training recipies, inference and performance
    visit: github.com/NVIDIA/DeepLearningExamples and/or ngc.nvidia.com
    Args (type[, default value]):
        pretrained (bool, True): If True, returns a model pretrained on LJ Speech dataset.
        model_math (str, 'fp32'): returns a model in given precision ('fp32' or 'fp16')
        n_symbols (int, 148): Number of symbols used in a sequence passed to the prenet, see
                              https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/SpeechSynthesis/Tacotron2/tacotron2/text/symbols.py
        p_attention_dropout (float, 0.1): dropout probability on attention LSTM (1st LSTM layer in decoder)
        p_decoder_dropout (float, 0.1): dropout probability on decoder LSTM (2nd LSTM layer in decoder)
        max_decoder_steps (int, 1000): maximum number of generated mel spectrograms during inference
    """

    from tacotron2 import model as tacotron2

    fp16 = "model_math" in kwargs and kwargs["model_math"] == "fp16"
    force_reload = "force_reload" in kwargs and kwargs["force_reload"]

    if pretrained:
        if fp16:
            checkpoint = 'https://api.ngc.nvidia.com/v2/models/nvidia/tacotron2_pyt_ckpt_amp/versions/19.09.0/files/nvidia_tacotron2pyt_fp16_20190427'
        else:
            checkpoint = 'https://api.ngc.nvidia.com/v2/models/nvidia/tacotron2_pyt_ckpt_fp32/versions/19.09.0/files/nvidia_tacotron2pyt_fp32_20190427'
        ckpt_file = _download_checkpoint(checkpoint, force_reload)
        ckpt = torch.load(ckpt_file)
        state_dict = ckpt['state_dict']
        if checkpoint_from_distributed(state_dict):
            state_dict = unwrap_distributed(state_dict)
        config = ckpt['config']
    else:
        config = {'mask_padding': False, 'n_mel_channels': 80, 'n_symbols': 148,
                  'symbols_embedding_dim': 512, 'encoder_kernel_size': 5,
                  'encoder_n_convolutions': 3, 'encoder_embedding_dim': 512,
                  'attention_rnn_dim': 1024, 'attention_dim': 128,
                  'attention_location_n_filters': 32,
                  'attention_location_kernel_size': 31, 'n_frames_per_step': 1,
                  'decoder_rnn_dim': 1024, 'prenet_dim': 256,
                  'max_decoder_steps': 1000, 'gate_threshold': 0.5,
                  'p_attention_dropout': 0.1, 'p_decoder_dropout': 0.1,
                  'postnet_embedding_dim': 512, 'postnet_kernel_size': 5,
                  'postnet_n_convolutions': 5, 'decoder_no_early_stopping': False}
        for k,v in kwargs.items():
            if k in config.keys():
                config[k] = v

    m = tacotron2.Tacotron2(**config)

    if pretrained:
        m.load_state_dict(state_dict)

    return m

def nvidia_tts_utils():
    
    class Processing:
        
        from tacotron2.text import text_to_sequence
        
        @staticmethod
        def pad_sequences(batch):
            # Right zero-pad all one-hot text sequences to max input length
            input_lengths, ids_sorted_decreasing = torch.sort(
                torch.LongTensor([len(x) for x in batch]),
                dim=0, descending=True)
            max_input_len = input_lengths[0]

            text_padded = torch.LongTensor(len(batch), max_input_len)
            text_padded.zero_()
            for i in range(len(ids_sorted_decreasing)):
                text = batch[ids_sorted_decreasing[i]]
                text_padded[i, :text.size(0)] = text

            return text_padded, input_lengths
        
        @staticmethod
        def prepare_input_sequence(texts, cpu_run=False):

            d = []
            for i,text in enumerate(texts):
                d.append(torch.IntTensor(
                    Processing.text_to_sequence(text, ['english_cleaners'])[:]))

            text_padded, input_lengths = Processing.pad_sequences(d)
            if not cpu_run:
                text_padded = text_padded.cuda().long()
                input_lengths = input_lengths.cuda().long()
            else:
                text_padded = text_padded.long()
                input_lengths = input_lengths.long()

            return text_padded, input_lengths
    
    return Processing()