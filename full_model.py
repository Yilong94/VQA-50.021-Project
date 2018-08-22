import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.utils.rnn import pack_padded_sequence
##################
from resnet import resnet as caffe_resnet
##################
import config


class Net(nn.Module):
    """ Re-implementation of ``Show, Ask, Attend, and Answer: A Strong Baseline For Visual Question Answering'' [0]

    [0]: https://arxiv.org/abs/1704.03162
    """

    def __init__(self, embedding_tokens):
        super(Net, self).__init__()
        question_features = 1024
        vision_features = config.output_features
        glimpses = 2

        self.text = LSTM_RNN(
            embedding_tokens=embedding_tokens,
            embedding_features=300,
            lstm_features=question_features,
            dropout=0.5,
        )

        self.average_pooling = SpatialAveragePooling(
        )

        self.classifier = Classifier(
            in_features= vision_features + question_features,
            mid_features=1024,
            out_features=config.max_answers,
            drop=0.5,
        )

        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                init.xavier_uniform(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, v, q, q_len):
        q = self.text(q, list(q_len.data))
        # print("Features")
        # print(v.shape)
        v = v / (v.norm(p=2, dim=1, keepdim=True).expand_as(v) + 1e-8)
        # print("L2 norm")
        # print(v.shape)
        v = self.average_pooling(v)
        # print("Average_pooling")
        # print(v.shape)
        # a = self.attention(v, q)
        # v = apply_attention(v, a)
        # print("Concat")
        v = torch.squeeze(v,-1)
        v = torch.squeeze(v,-1)
        # print(v.shape,q.shape)
        combined = torch.cat([v, q], dim=1)
        # print("Classifier")
        # print(combined.shape)
        answer = self.classifier(combined)
        return answer




class SpatialAveragePooling(nn.Module):
    """docstring for SpatialAveragePooling"""
    def __init__(self):
        super(SpatialAveragePooling, self).__init__()
        self.ave_pool = torch.nn.AvgPool2d((14,14), stride=None, padding=0, ceil_mode=False, count_include_pad=True)
        

    def forward(self,x):
        return self.ave_pool(x)



class Classifier(nn.Sequential):
    def __init__(self, in_features, mid_features, out_features, drop=0.0):
        super(Classifier, self).__init__()
        self.add_module('drop1', nn.Dropout(drop))
        self.add_module('lin1', nn.Linear(in_features, mid_features))
        self.add_module('relu', nn.ReLU())
        self.add_module('drop2', nn.Dropout(drop))
        self.add_module('lin2', nn.Linear(mid_features, out_features))


class LSTM_RNN(nn.Module):
    def __init__(self, embedding_tokens, embedding_features=300, lstm_features=1024, dropout=0.5):
        super(LSTM_RNN, self).__init__()
        # num_embeddings (int) - size of the dictionary of embeddings
        # embedding_dim (int) - the size of each embedding vector
        # padding_idx (int, optional) - If given, pads the output with the 
        # embedding vector at padding_idx (initialized to zeros) whenever it encounters the index.
        self.embedding = nn.Embedding(embedding_tokens, embedding_features, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        self.tanh = nn.Tanh()
        self.lstm = nn.LSTM(input_size=embedding_features,
                            hidden_size=lstm_features,
                            num_layers=1
                            )
        self.features = lstm_features

        # initialize lstm weights to normal distribution and bias to zero 
        self.init_lstm_weights(self.lstm.weight_ih_l0)
        self.init_lstm_weights(self.lstm.weight_hh_l0)
        self.lstm.bias_ih_l0.data.zero_()
        self.lstm.bias_hh_l0.data.zero_()

        # initialize embedding weights
        init.normal_(self.embedding.weight)

    def init_lstm_weights(self, weight):
        for w in weight.chunk(4,0):
            init.normal_(w)

    def forward(self, question_tensor, question_tensor_length):
        # print(question_tensor.shape)
        question_embedded = self.embedding(question_tensor)
        # print(question_embedded.shape)
        question_tanh = self.tanh(self.dropout(question_embedded))
        question_packed = pack_padded_sequence(question_tanh, question_tensor_length, batch_first=True)
        _, (_, c) = self.lstm(question_packed)
        return c.squeeze(0)


# class TextProcessor(nn.Module):
#     def __init__(self, embedding_tokens, embedding_features, lstm_features, drop=0.0):
#         super(TextProcessor, self).__init__()
#         self.embedding = nn.Embedding(embedding_tokens, embedding_features, padding_idx=0)
#         self.drop = nn.Dropout(drop)
#         self.tanh = nn.Tanh()
#         self.lstm = nn.LSTM(input_size=embedding_features,
#                             hidden_size=lstm_features,
#                             num_layers=1)
#         self.features = lstm_features

#         self._init_lstm(self.lstm.weight_ih_l0)
#         self._init_lstm(self.lstm.weight_hh_l0)
#         self.lstm.bias_ih_l0.data.zero_()
#         self.lstm.bias_hh_l0.data.zero_()

#         init.xavier_uniform(self.embedding.weight)

#     def _init_lstm(self, weight):
#         for w in weight.chunk(4, 0):
#             init.xavier_uniform(w)

#     def forward(self, q, q_len):
#         embedded = self.embedding(q)
#         tanhed = self.tanh(self.drop(embedded))
#         packed = pack_padded_sequence(tanhed, q_len, batch_first=True)
#         _, (_, c) = self.lstm(packed)
#         return c.squeeze(0)
