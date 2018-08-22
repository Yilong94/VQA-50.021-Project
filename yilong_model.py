import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.utils.rnn import pack_padded_sequence

class Net(nn.Module):
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
		print(question_tensor.shape)
		question_embedded = self.embedding(question_tensor)
		print(question_embedded.shape)
		question_tanh = self.tanh(self.dropout(question_embedded))
		question_packed = pack_padded_sequence(question_tanh, question_tensor_length, batch_first=True)
		_, (_, c) = self.lstm(question_packed)
		return c.squeeze(0)

# if __name__=='__main__':
	# no. of vocab words = 20
	# lstm_rnn = LSTM_RNN(20)
	# # batch of 3 sentences
	# # value represents the index of the vocab word in embedding
	# # note that index cannot exceed that of the length of vocab, which is 20 in this case
	# sentence_tensor_batch = torch.LongTensor(
	# 						[[19,1,3,5,6,6],
	# 						[1,2,3,4,5,6],
	# 						[1,2,3,4,0,0]]
	# 						)
	# # test output of lstm
	# print(lstm_rnn(sentence_tensor_batch, [6,6,4]))
	# # note: outputs are randomized due to the random initialization of weights using normal dist




