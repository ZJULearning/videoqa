import theano
import theano.tensor as T
from blocks.bricks import Linear, Softmax
from blocks.initialization import Constant, IsotropicGaussian
from blocks.bricks import recurrent
from blocks.bricks.recurrent import LSTM, Bidirectional

class visualEncoder:
    def __init__(self, visual_dim, hidden_dim):
        self.forward_lstm= LSTM(hidden_dim,
                                name='visual_forward_lstm',
                                weights_init=IsotropicGaussian(0.01),
                                biases_init=Constant(0))
        self.backward_lstm= LSTM(hidden_dim,
                                 name='visual_backward_lstm',
                                 weights_init=IsotropicGaussian(0.01),
                                 biases_init=Constant(0))
        self.x_to_h_forward = Linear(visual_dim,
                                     hidden_dim * 4,
                                     name='visual_forward_x_to_h',
                                     weights_init=IsotropicGaussian(0.01),
                                     biases_init=Constant(0))
        self.x_to_h_backward = Linear(visual_dim,
                                      hidden_dim * 4,
                                      name='visual_backward_x_to_h',
                                      weights_init=IsotropicGaussian(0.01),
                                      biases_init=Constant(0))

        self.forward_lstm.initialize()
        self.backward_lstm.initialize()
        self.x_to_h_forward.initialize()
        self.x_to_h_backward.initialize()

    # fixed video_length
    # frames: batch_size x video_length x visual_dim
    def apply(self, frames):
        Wx = self.x_to_h_forward.apply(frames)
        Wx_r = self.x_to_h_backward.apply(frames[:, ::-1, :])
        # video_length x batch_size x hidden_dim
        Wx = Wx.swapaxes(0, 1)
        Wx_r = Wx_r.swapaxes(0, 1)
        # nSteps x batch size x dim 
        hf, cf = self.forward_lstm.apply(Wx)
        hb, cb = self.backward_lstm.apply(Wx_r)
        # video_length x batch_size x (2 x hidden_dim)
        h = T.concatenate([hf, hb[::-1]], axis=2)
        # batch_size x video_length x (2 x hidden_dim)
        return h.swapaxes(0, 1)
