import theano
import theano.tensor as T
from blocks.bricks import Linear, Softmax
from blocks.initialization import Constant, IsotropicGaussian


class embeddingLayer:
    def __init__(self, word_dim, visual_dim, joint_dim):
        self.word_embed = Linear(word_dim,
                                 joint_dim,
                                 name='word_to_joint',
                                 weights_init=IsotropicGaussian(0.01),
                                 biases_init=Constant(0))
        self.visual_embed = Linear(visual_dim,
                                   joint_dim,
                                   name='visual_to_joint',
                                   weights_init=IsotropicGaussian(0.01),
                                   biases_init=Constant(0))
        self.word_embed.initialize()
        self.visual_embed.initialize()

    # words: batch_size x q x word_dim
    # video: batch_size x video_length x visual_dim
    def apply(self, words, video, u1, u2):
        w = self.word_embed.apply(words)
        v = self.visual_embed.apply(video)
        w = T.tanh(w)
        v = T.tanh(v)
        u = T.concatenate([u1, u2], axis=1)
        u = self.word_embed.apply(u)
        return w, v, u

    def apply_sentence(self, words, u1, u2):
        w = self.word_embed.apply(words)
        w = T.tanh(w)
        u = T.concatenate([u1, u2], axis=1)
        u = self.word_embed.apply(u)
        return w, u
