import copy
import torch.nn as nn
import pickle5 as pickle
from networks import MultiHeadedAttention, PositionalEncoding, PositionwiseFeedForward
from networks import Encoder, EncoderDecoder, EncoderLayer
from networks import Decoder, DecoderLayer
from networks import Generator, Embeddings


class TLogModel:
    def __init__(self, src_vocab, tgt_vocab, n_layers=3,
                 in_features=512, out_features=2048, num_heads=8, dropout=0.1, max_len=20, weights_path=None):
        """Construct a model from hyper parameters.
        Parameters
        ----------
        src_vocab : int
            Length of source vocabulary.
        tgt_vocab : int
            Length of target vocabulary
        n_layers : int
            Number of encoder and decoder layers.
        in_features : int
            number of input features
        out_features : int
            number of output features
        dropout : float
            Dropout weights percentage
        max_len : int
        num_heads : int
            Number of heads for the multi-head model


        """
        c = copy.deepcopy
        attn = MultiHeadedAttention(num_heads, in_features)
        ff = PositionwiseFeedForward(in_features, out_features, dropout)
        position = PositionalEncoding(in_features, dropout, max_len)
        self.model = EncoderDecoder(
            Encoder(EncoderLayer(in_features, c(attn), c(ff), dropout), n_layers),
            Decoder(DecoderLayer(in_features, c(attn), c(attn), c(ff), dropout), n_layers),
            nn.Sequential(Embeddings(in_features, src_vocab), c(position)),
            nn.Sequential(Embeddings(in_features, tgt_vocab), c(position)),
            Generator(in_features, tgt_vocab))
        self._update_model_weights(weights_path)
        # This was important from their code.
        # Initialize parameters with Glorot / fan_avg.
        if weights_path == None:
            for p in self.model.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

    def get_model(self):
        return self.model

    def _update_model_weights(self, weights_path):
        if weights_path is not None:
            with open(weights_path, 'rb') as file:
                weights = pickle.load(file)

            self.model.state_dict().update(weights)
            model_dict = self.model.state_dict()
            pretrained_dict = {k: v for k, v in weights.items() if k in model_dict}
            model_dict.update(pretrained_dict) 
            self.model.load_state_dict(model_dict)
