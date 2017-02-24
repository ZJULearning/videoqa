import theano
import theano.tensor as T
from blocks.bricks import Linear, Softmax
from blocks.initialization import Constant, IsotropicGaussian
from blocks.bricks.recurrent import GatedRecurrent, LSTM
from theano.tensor.nnet import relu
from theano.tensor.nnet.nnet import sigmoid

class seqDecoder:
    def __init__(self, feature_dim, memory_dim, fc1_dim, fc2_dim):
        self.W = Linear(input_dim=feature_dim,
                        output_dim=memory_dim * 4,
                        weights_init=IsotropicGaussian(0.01),
                        biases_init=Constant(0),
                        use_bias=False,
                        name='seqDecoder_W')
        self.GRU_A = LSTM(feature_dim,
                          name='seqDecoder_A',
                          weights_init=IsotropicGaussian(0.01),
                          biases_init=Constant(0))
        self.GRU_B = LSTM(memory_dim,
                          name='seqDecoder_B',
                          weights_init=IsotropicGaussian(0.01),
                          biases_init=Constant(0))
        self.W.initialize()
        self.GRU_A.initialize()
        self.GRU_B.initialize()
        self.fc1 = Linear(input_dim=memory_dim,
                          output_dim=fc1_dim,
                          weights_init=IsotropicGaussian(0.01),
                          biases_init=Constant(0),
                          name='fc1')
        self.fc2 = Linear(input_dim=fc1_dim,
                          output_dim=fc2_dim,
                          weights_init=IsotropicGaussian(0.01),
                          biases_init=Constant(0),
                          name='fc2')

        self.fc1.initialize()
        self.fc2.initialize()

    # A: the encoding of GRU_A,
    # B: the encoding of GRU_B
    # padding: the tensor constant
    def apply(self, output_length, A, B, padding):
        A_, garbage = self.GRU_A.apply(padding, states=A)
        WA_ = self.W.apply(A_)
        # output_length x batch_size x output_dim
        B_, garbage = self.GRU_B.apply(WA_, states=B)
        # batch_size x output_length x output_dim
        B_ =  B_.swapaxes(0,1)
        fc1_r = relu(self.fc1.apply(B_))
        fc2_r = relu(self.fc2.apply(fc1_r))
        return fc2_r
