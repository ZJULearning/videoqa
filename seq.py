import theano
import theano.tensor as T
from blocks.bricks import Linear, Softmax
from blocks.initialization import Constant
from blocks.bricks import recurrent
from blocks.bricks.recurrent import LSTM, Bidirectional
from gif_encoder import *
from rereader_seq import *
from question_encoder import *
from rewatcher_seq import *
from embedding_layer import *
from blocks.bricks.cost import CategoricalCrossEntropy
from theano.tensor.extra_ops import to_one_hot      
from decoder import *

class seq:
    def __init__(self, batch_size, output_length,
                 visual_dim, word_dim,
                 visual_feature_dim,
                 question_feature_dim,
                 joint_dim,
                 memory_dim,
                 output_dim,
                 fc1_dim,
                 fc2_dim,
                 voc_size):
        # the video encoder
        self.video_encoder = visualEncoder(
            visual_dim,
            visual_feature_dim)
        self.sentence_encoder = questionEncoder(
            word_dim,
            question_feature_dim)
        self.toJoint = embeddingLayer(
            2 * question_feature_dim,
            2 * visual_feature_dim,
            joint_dim)
        self.rewatcher = impatientLayer(
            joint_dim,
            memory_dim,
            output_dim)
        self.rereader = iwLayer(
            joint_dim,
            memory_dim,
            output_dim)
        self.seq_gen = seqDecoder(
            joint_dim,
            output_dim,
            fc1_dim,
            fc2_dim)
        self.softmax_layer = Softmax()
        self.bs = batch_size
        self.output_length = output_length
        self.voc_size = voc_size
                                           

    def build_model(self, frame, q, q_rev, mask, maskMat, mask01, padding):
        bs = self.bs
        # visual dim -> visual feature dim
        video_embedding = self.video_encoder.apply(frame)
        # wod_dim -> question feature dimA
        question_embedding, u1, u2 = self.sentence_encoder.apply(q, q_rev, mask, bs)
        # -> joint_dim
        questionJoint, videoJoint, u = self.toJoint.apply(words=question_embedding,
                                                          video=video_embedding,
                                                          u1=u1,
                                                          u2=u2)
        question = questionJoint[:, -1, :]
        video = videoJoint[:, -1, :]
        # bs x joint_dim, bs x output_dim

        fc_r = self.seq_gen.apply(self.output_length, video, question, padding)
        fc = fc_r.reshape((self.bs*self.output_length, self.voc_size))
        self.softmax_result = self.softmax_layer.apply(fc)
        self.pred = T.argmax(self.softmax_result, axis=1)
        self.pred = self.pred.reshape((self.bs, self.output_length))

    # groundtruth_: batch_size x output_length
    # mask_01: (batch_size x output_length)
    # this mask is a 0-1 matrix where 0 indicates padding area of the answer
    def loss(self, groundtruth_, mask_01):
        mask = mask_01.flatten()
        gt = groundtruth_.flatten()
        
        self.p = self.softmax_result[T.arange(self.bs * self.output_length),
                                     gt]
        self.cost_ = T.log(self.p + 1e-20)
        self.cost = -T.sum(self.cost_ * mask) / self.bs
        self.cost.name = 'softmax_cost'
        return self.cost

    def error(self, groundtruth, mask_01):
        return T.neq(T.sum(T.neq(self.pred, groundtruth) * mask_01, axis=1), 0).sum() / self.bs

    def predict(self):
        return self.pred
