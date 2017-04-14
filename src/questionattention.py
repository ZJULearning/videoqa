# impatient watcher
import theano
import theano.tensor as T
from blocks.bricks import Linear, Softmax
from blocks.initialization import Constant, IsotropicGaussian
from blocks.bricks.recurrent import GatedRecurrent, LSTM

class questionAttentionLayer:
    def __init__(self, feature_dim, hidden_dim, output_dim):
        self.image_embed = Linear(input_dim=feature_dim,
                                  output_dim=hidden_dim,
                                  weights_init=IsotropicGaussian(0.01),
                                  biases_init=Constant(0),
                                  use_bias=False,
                                  name='iw_image_embed')
        self.word_embed = Linear(input_dim=feature_dim,
                                 output_dim=hidden_dim,
                                 weights_init=IsotropicGaussian(0.01),
                                 biases_init=Constant(0),
                                 use_bias=False,
                                 name='iw_word_embed')
        self.r_embed = Linear(input_dim=feature_dim,
                              output_dim=hidden_dim,
                              weights_init=IsotropicGaussian(0.01),
                              biases_init=Constant(0),
                              use_bias=False,
                              name='iw_r_embed')
        self.m_to_s = Linear(input_dim=hidden_dim,
                             output_dim=1,
                             weights_init=IsotropicGaussian(0.01),
                             biases_init=Constant(0),
                             use_bias=False,
                             name='iw_m_to_s')
        self.attention_dist = Softmax(name='iw_attetion')
        self.r_to_r = Linear(input_dim=feature_dim,
                             output_dim=feature_dim,
                             weights_init=IsotropicGaussian(0.01),
                             biases_init=Constant(0),
                             use_bias=False,
                             name='iw_r_to_r')
        # self.r_to_g = Linear(input_dim=feature_dim,
        #                      output_dim=output_dim,
        #                      weights_init=IsotropicGaussian(0.01),
        #                      biases_init=Constant(0),
        #                      use_bias=False,
        #                      name='iw_r_to_g')
        self.image_embed.initialize()
        self.word_embed.initialize()
        self.r_embed.initialize()
        self.m_to_s.initialize()
        self.r_to_r.initialize()
        # self.r_to_g.initialize()
        self.seq = LSTM(feature_dim,
                        name='rereader_seq',
                        weights_init=IsotropicGaussian(0.01),
                        biases_init=Constant(0))
        self.seq_embed = Linear(feature_dim,
                                output_dim * 4,
                                name='rereader_seq_embed',
                                weights_init=IsotropicGaussian(0.01),
                                biases_init=Constant(0),
                                use_bias=False)

        self.seq.initialize()
        self.seq_embed.initialize()


    # video: batch_size x video_length x feature_dim
    # query: batch_size x q x feature_dim
    # mask: this mask is different from other masks
    # batch_size x q
    # eg.
    # -10000 == -np.Inf
    # 1:   0, 0, 0, 0, 0, -10000, -10000, -10000
    # 2:   0, 0, 0, 0, -10000, -10000, -10000
    # 3:   0, 0, 0, 0, 0, 0, 0, -10000
    def apply(self, video, query, mask, batch_size):
        # batch_size x q x hidden_dim
        att1 = self.word_embed.apply(query)


        # batch_size x hidden_dim
        y_q = query
        att3 = self.image_embed.apply(video)
        att = att1 + att3.dimshuffle(0, 'x', 1)
        # batch_size x q x hidden_dim
        m = T.tanh(att)
        # batch_size x q
        s = self.m_to_s.apply(m)
        s = s.reshape((s.shape[0], s.shape[1]))
        # ignore the question padding 0s
        s = s + mask
        s = self.attention_dist.apply(s)
        y_q_s = y_q.swapaxes(1, 2)
        r = T.batched_dot(y_q_s, s) 

        # batch_size x feature_dim
        return r
