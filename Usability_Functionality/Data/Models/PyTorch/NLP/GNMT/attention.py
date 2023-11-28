import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

class BahdanauAttention(nn.Module):
    def __init__(self, query_size, key_size, num_units, normalize=False,
                 batch_first=False, init_weight=0.1):
        super(BahdanauAttention, self).__init__()
        self.normalize = normalize
        self.batch_first = batch_first
        self.num_units = num_units
        self.linear_q = nn.Linear(query_size, num_units, bias=False)
        self.linear_k = nn.Linear(key_size, num_units, bias=False)
        nn.init.uniform_(self.linear_q.weight.data, -init_weight, init_weight)
        nn.init.uniform_(self.linear_k.weight.data, -init_weight, init_weight)
        self.linear_att = Parameter(torch.Tensor(num_units))
        self.mask = None
        if self.normalize:
            self.normalize_scalar = Parameter(torch.Tensor(1))
            self.normalize_bias = Parameter(torch.Tensor(num_units))
        else:
            self.register_parameter('normalize_scalar', None)
            self.register_parameter('normalize_bias', None)
        self.reset_parameters(init_weight)

    def reset_parameters(self, init_weight):
        stdv = 1. / math.sqrt(self.num_units)
        self.linear_att.data.uniform_(-init_weight, init_weight)
        if self.normalize:
            self.normalize_scalar.data.fill_(stdv)
            self.normalize_bias.data.zero_()

    def set_mask(self, context_len, context):
        if self.batch_first:
            max_len = context.size(1)
        else:
            max_len = context.size(0)
        indices = torch.arange(0, max_len, dtype=torch.int64,
                               device=context.device)
        self.mask = indices >= (context_len.unsqueeze(1))

    def calc_score(self, att_query, att_keys):
        b, t_k, n = att_keys.size()
        t_q = att_query.size(1)
        att_query = att_query.unsqueeze(2).expand(b, t_q, t_k, n)
        att_keys = att_keys.unsqueeze(1).expand(b, t_q, t_k, n)
        sum_qk = att_query + att_keys
        if self.normalize:
            sum_qk = sum_qk + self.normalize_bias
            linear_att = self.linear_att / self.linear_att.norm()
            linear_att = linear_att * self.normalize_scalar
        else:
            linear_att = self.linear_att
        out = torch.tanh(sum_qk).matmul(linear_att)
        return out

    def forward(self, query, keys):
        if not self.batch_first:
            keys = keys.transpose(0, 1)
            if query.dim() == 3:
                query = query.transpose(0, 1)
        if query.dim() == 2:
            single_query = True
            query = query.unsqueeze(1)
        else:
            single_query = False
        b = query.size(0)
        t_k = keys.size(1)
        t_q = query.size(1)
        processed_query = self.linear_q(query)
        processed_key = self.linear_k(keys)
        scores = self.calc_score(processed_query, processed_key)
        if self.mask is not None:
            mask = self.mask.unsqueeze(1).expand(b, t_q, t_k)
            scores.masked_fill_(mask, -65504.0)
        scores_normalized = F.softmax(scores, dim=-1)
        context = torch.bmm(scores_normalized, keys)
        if single_query:
            context = context.squeeze(1)
            scores_normalized = scores_normalized.squeeze(1)
        elif not self.batch_first:
            context = context.transpose(0, 1)
            scores_normalized = scores_normalized.transpose(0, 1)
        return context, scores_normalized
