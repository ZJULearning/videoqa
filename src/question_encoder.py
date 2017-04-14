import theano
import theano.tensor as T
from blocks.bricks import Linear, Softmax
from blocks.initialization import Constant, IsotropicGaussian
from blocks.bricks import recurrent
from blocks.bricks.recurrent import LSTM, Bidirectional

class questionEncoder:
    def __init__(self, word_dim, hidden_dim):
        self.forward_lstm= LSTM(hidden_dim,
                                name='question_forward_lstm',
                                weights_init=IsotropicGaussian(0.01),
                                biases_init=Constant(0))
        self.backward_lstm= LSTM(hidden_dim,
                                 name='question_backward_lstm',
                                 weights_init=IsotropicGaussian(0.01),
                                 biases_init=Constant(0))
        self.x_to_h_forward = Linear(word_dim,
                                     hidden_dim * 4,
                                     name='word_x_to_h_forward',
                                     weights_init=IsotropicGaussian(0.01),
                                     biases_init=Constant(0))
        self.x_to_h_backward = Linear(word_dim,
                                      hidden_dim * 4,
                                      name='word_x_to_h_backward',
                                      weights_init=IsotropicGaussian(0.01),
                                      biases_init=Constant(0))

        self.forward_lstm.initialize()
        self.backward_lstm.initialize()
        self.x_to_h_forward.initialize()
        self.x_to_h_backward.initialize()

    # variable question length
    # words: batch_size x q x word_dim
    # words_reverse: be the reverse sentence of words
    #                padding with 0 to max length q
    # mask: batch_size 
    def apply(self, words, words_reverse, mask_, batch_size):
        mask = mask_.flatten()
        # batch_size x q x hidden_dim
        Wx = self.x_to_h_forward.apply(words)
        Wx_r = self.x_to_h_backward.apply(words_reverse)
        # q x batch_size x hidden_dim
        Wx = Wx.swapaxes(0, 1)
        Wx_r = Wx_r.swapaxes(0, 1)
        # q x batch_size x hidden_dim
        hf, cf = self.forward_lstm.apply(Wx)
        hb, cb = self.backward_lstm.apply(Wx_r)
        for i in range(batch_size):
            T.set_subtensor(hb[0:mask[i]+1, i, :], hb[0:mask[i]+1, i, :][::-1])

        # q x batch_size x (2 x hidden_dim)
        h = T.concatenate([hf, hb], axis=2)
        # batch_size x hidden_dim
        y_q = hf[mask, range(batch_size), :]
        y_1 = hb[0, range(batch_size), :]
        
        return h.swapaxes(0, 1), y_q, y_1
