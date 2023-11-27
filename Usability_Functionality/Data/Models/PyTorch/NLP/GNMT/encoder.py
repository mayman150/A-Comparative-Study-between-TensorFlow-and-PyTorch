import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence

import seq2seq.data.config as config
from seq2seq.utils import init_lstm_

class ResidualRecurrentEncoder(nn.Module):
    def __init__(self, vocab_size, hidden_size=1024, num_layers=4, dropout=0.2,
                 batch_first=False, embedder=None, init_weight=0.1):
        super(ResidualRecurrentEncoder, self).__init__()
        self.batch_first = batch_first
        self.rnn_layers = nn.ModuleList()
        self.rnn_layers.append(
            nn.LSTM(hidden_size, hidden_size, num_layers=1, bias=True,
                    batch_first=batch_first, bidirectional=True))
        self.rnn_layers.append(
            nn.LSTM((2 * hidden_size), hidden_size, num_layers=1, bias=True,
                    batch_first=batch_first))
        for _ in range(num_layers - 2):
            self.rnn_layers.append(
                nn.LSTM(hidden_size, hidden_size, num_layers=1, bias=True,
                        batch_first=batch_first))
        for lstm in self.rnn_layers:
            init_lstm_(lstm, init_weight)

        self.dropout = nn.Dropout(p=dropout)

        if embedder is not None:
            self.embedder = embedder
        else:
            self.embedder = nn.Embedding(vocab_size, hidden_size,
                                         padding_idx=config.PAD)
            nn.init.uniform_(self.embedder.weight.data, -init_weight,
                             init_weight)

    def forward(self, inputs, lengths):
        x = self.embedder(inputs)
        x = self.dropout(x)
        x = pack_padded_sequence(x, lengths.cpu().numpy(),
                                 batch_first=self.batch_first)
        x, _ = self.rnn_layers[0](x)
        x, _ = pad_packed_sequence(x, batch_first=self.batch_first)
        x = self.dropout(x)
        x, _ = self.rnn_layers[1](x)
        for i in range(2, len(self.rnn_layers)):
            residual = x
            x = self.dropout(x)
            x, _ = self.rnn_layers[i](x)
            x = x + residual

        return x
