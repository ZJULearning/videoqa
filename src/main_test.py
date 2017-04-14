from forgettable import *
import theano
import theano.tensor as T
import numpy as np
from blocks.graph import ComputationGraph
import h5py
from fuel.datasets.hdf5 import H5PYDataset
from fuel.transformers import Flatten
from fuel.streams import DataStream
from fuel.schemes import SequentialScheme, ShuffledScheme
from blocks.extensions.monitoring import DataStreamMonitoring
from blocks.algorithms import GradientDescent, CompositeRule, StepClipping, RMSProp, Adam
from blocks.main_loop import MainLoop
from blocks.extensions import FinishAfter, Printing
from blocks.extensions.monitoring import DataStreamMonitoring, TrainingDataMonitoring
from blocks.extensions import FinishAfter, Timing, Printing, ProgressBar
from blocks.monitoring import aggregation
from blocks.model import Model
from blocks.extensions.saveload import Checkpoint
from blocks_extras.extensions.plot import Plot
from saveSnapshot import *

prob_dim = 1063
bs = 100
# max ans length + <EOS> + <Unknown>
max_ans_length = 22
joint_dim = 1024
model = forgettable(
    bs, max_ans_length,
    4096, 300,
    2048,
    1024,
    1024,
    1024,
    1024,
    2048,
    1063,
    1063)
video_length = 300
max_len = 27
visual_dim = 4096
word_dim = 300
frames = theano.shared(np.asarray(np.zeros((bs,
                                            video_length,
                                            visual_dim)),
                                  dtype=theano.config.floatX),
                       borrow=True,
                       name='visual_features')

qas = theano.shared(np.asarray(np.zeros((bs,
                                         max_len,
                                         word_dim)),
                               dtype=theano.config.floatX),
                    borrow=True,
                    name='question_features')

qas_rev = theano.shared(np.asarray(np.zeros((bs,
                                             max_len,
                                             word_dim)),
                                   dtype=theano.config.floatX),
                        borrow=True,
                        name='question_features_reverse')

mask = theano.shared(np.asarray(np.ones((bs)),
                                dtype='int'),
                     borrow=True,
                     name='mask')

maskMat = theano.shared(np.asarray(np.zeros((bs, max_len)),
                                   dtype=theano.config.floatX),
                        borrow=True,
                        name='mask_matrix')


padding = T.constant(np.zeros((max_ans_length,
                               bs,
                               joint_dim * 4)).astype(np.float32))
m = np.ones((bs, max_ans_length))
m[:, 5::] = 0
mask01 = theano.shared(m.astype(np.float32))

gt = T.lmatrix('label')
model.build_model(frames, qas, qas_rev, mask, maskMat, mask01, padding)

cost = model.loss(gt, mask01)
cost.name = 'cost'

gt_ = np.zeros((bs, max_ans_length)).astype(np.int64)
f = theano.function([gt], cost)

print(f(gt_))
