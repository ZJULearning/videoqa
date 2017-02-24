import theano
import theano.tensor as T
from blocks.bricks import Linear, Softmax
from blocks.initialization import Constant, IsotropicGaussian
from blocks.bricks.recurrent import GatedRecurrent, LSTM
class impatientLayer:
    # both visual and word feature are in the joint space
    # of dim: feature_dim
    # hidden_dim: dim of m
    # output_dim: final joint document query representation dim
    def __init__(self, feature_dim, hidden_dim, output_dim):
        self.image_embed = Linear(input_dim=feature_dim,
                                  output_dim=hidden_dim,
                                  weights_init=IsotropicGaussian(0.01),
                                  biases_init=Constant(0),
                                  use_bias=False,
                                  name='image_embed')
        self.word_embed = Linear(input_dim=feature_dim,
                                 output_dim=hidden_dim,
                                 weights_init=IsotropicGaussian(0.01),
                                 biases_init=Constant(0),
                                 use_bias=False,
                                 name='word_embed')
        self.r_embed = Linear(input_dim=feature_dim,
                              output_dim=hidden_dim,
                              weights_init=IsotropicGaussian(0.01),
                              biases_init=Constant(0),
                              use_bias=False,
                              name='r_embed')
        self.m_to_s = Linear(input_dim=hidden_dim,
                             output_dim=1,
                             weights_init=IsotropicGaussian(0.01),
                             biases_init=Constant(0),
                             use_bias=False,
                             name='m_to_s')
        self.attention_dist = Softmax(name='attention_dist_softmax')
        self.r_to_r = Linear(input_dim=feature_dim,
                             output_dim=feature_dim,
                             weights_init=IsotropicGaussian(0.01),
                             biases_init=Constant(0),
                             use_bias=False,
                             name='r_to_r')
        # self.r_to_g = Linear(input_dim=feature_dim,
        #                      output_dim=output_dim,
        #                      weights_init=IsotropicGaussian(0.01),
        #                      biases_init=Constant(0),
        #                      use_bias=False,
        #                      name='r_to_g')
        self.image_embed.initialize()
        self.word_embed.initialize()
        self.r_embed.initialize()
        self.m_to_s.initialize()
        self.r_to_r.initialize()
        # self.r_to_g.initialize()

        # the sequence to sequence LSTM
        self.seq = LSTM(output_dim,
                        name='rewatcher_seq',
                        weights_init=IsotropicGaussian(0.01),
                        biases_init=Constant(0))
        self.seq_embed = Linear(feature_dim,
                                output_dim * 4,
                                name='rewatcher_seq_embed',
                                weights_init=IsotropicGaussian(0.01),
                                biases_init=Constant(0),
                                use_bias=False)

        self.seq.initialize()
        self.seq_embed.initialize()

    # doc: row major batch_size x doc_length x feature_dim
    # query: row major batch_size x q x feature_dim
    # mask: mask of query batch_size
    # mask: length of a sentence - 1
    def apply(self, doc, query, mask_, batch_size):
        # batch_size x doc_length x hidden_dim
        mask = mask_.flatten()
        att1 = self.image_embed.apply(doc)

        # y_q_i: the ith token of question
        #        batch_size x feature_dim
        # r_1: r_m_1
        #        batch_size x feature_dim
        # y_d: document
        #        batch_size x doc_length x feature_dim
        # y_d_m: d-to-m
        #        batch_size x doc_length x hidden_dim
        def one_step(y_q_i, r_1, y_d, y_d_m):
            # batch_size x hidden_dim
            att2 = self.r_embed.apply(r_1)
            # batch_size x hidden_dim
            att3 = self.word_embed.apply(y_q_i)
            att = y_d_m + att2.dimshuffle(0, 'x', 1) + att3.dimshuffle(0, 'x', 1)
            # batch_size x doc_length x hidden_dim
            m = T.tanh(att)
            # batch_size x doc_length x 1
            s = self.m_to_s.apply(m)
            # batch_size x doc_length
            s = s.reshape((s.shape[0], s.shape[1]))
            s = self.attention_dist.apply(s)
            y_d_s = y_d.swapaxes(1, 2)
            # return batch_size x feature_dim
            return T.batched_dot(y_d_s, s) + T.tanh(self.r_to_r.apply(r_1))

        # query: batch_size x q x feature_dim
        # r: q x batch_size x feature_dim
        r, updates = theano.scan(fn=one_step,
                                 sequences=[query.swapaxes(0,1)],
                                 outputs_info=T.zeros_like(doc[:, 0, :]),
                                 non_sequences=[doc, att1],
                                 n_steps=query.shape[1],
                                 name='impatient layer')

        # for the sequence encoder
        # q x batch_size x output_dim
        Wr = self.seq_embed.apply(r)
        # q x batch_size x output_dim
        seq_r, garbage = self.seq.apply(Wr)
        # batch_size x feature_dim
        r_q = r[mask, T.arange(batch_size), :]
        seq_r_q = seq_r[mask, T.arange(batch_size), :]
        # batch_size x output_dim
        return r_q, seq_r_q

