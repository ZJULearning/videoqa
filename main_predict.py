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
from blocks.extensions.saveload import Checkpoint, Load
from blocks_extras.extensions.plot import Plot
from saveSnapshot import *
from predictDataStream import *

prob_dim = 3591
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
    3591,
    3591)
video_length = 300
max_len = 27
visual_dim = 4096
word_dim = 300
# frames = theano.shared(np.asarray(np.zeros((bs,
#                                             video_length,
#                                             visual_dim)),
#                                   dtype=theano.config.floatX),
#                        borrow=True,
#                        name='visual_features')

# qas = theano.shared(np.asarray(np.zeros((bs,
#                                          max_len,
#                                          word_dim)),
#                                dtype=theano.config.floatX),
#                     borrow=True,
#                     name='question_features')

# qas_rev = theano.shared(np.asarray(np.zeros((bs,
#                                              max_len,
#                                              word_dim)),
#                                    dtype=theano.config.floatX),
#                         borrow=True,
#                         name='question_features_reverse')

# mask = theano.shared(np.asarray(np.ones((bs)),
#                                 dtype='int'),
#                      borrow=True,
#                      name='mask')

# maskMat = theano.shared(np.asarray(np.zeros((bs, max_len)),
#                                    dtype=theano.config.floatX),
#                         borrow=True,
#                         name='mask_matrix')


padding = T.constant(np.zeros((max_ans_length,
                               bs,
                               joint_dim * 4)).astype(np.float32))
# m = np.ones((bs, max_ans_length))
# m[:, 5::] = 0
# mask01 = theano.shared(m.astype(np.float32))
frames = T.tensor3('visual_features')
qas = T.tensor3('question_features')
qas_rev = T.tensor3('question_features_reverse')
mask = T.lmatrix('mask')
maskMat = T.matrix('mask_matrix')
mask01 = T.matrix('mask01')
gt = T.lmatrix('label')
model.build_model(frames, qas, qas_rev, mask, maskMat, mask01, padding)

cost = model.loss(gt, mask01)
cost.name = 'cost'
error = model.error(gt, mask01)
error.name = 'error'
result = model.predict()

cg = ComputationGraph(cost)

data_train = H5PYDataset('/home/xuehongyang/TGIF_open_161215.hdf5', which_sets=('train',),
                         subset=slice(0, 230689//bs*bs))

data_test = H5PYDataset('/home/xuehongyang/TGIF_open_161215.hdf5', which_sets=('test',),
                        sources=('question_features', 'question_features_reverse',
                                 'mask', 'mask_matrix', 'visual_features', ),
                        subset=slice(0, 32378//bs*bs))

data_stream_train = DataStream.default_stream(
        data_train,
        iteration_scheme=ShuffledScheme(data_train.num_examples, batch_size=bs))

data_stream_test = DataStream.default_stream(
        data_test,
        iteration_scheme=SequentialScheme(
                    data_test.num_examples, batch_size=bs))

learning_rate = 0.0002
n_epochs=100
algorithm = GradientDescent(cost=cost,
                            parameters=cg.parameters,
                            step_rule=CompositeRule([
                                StepClipping(10.),
                                Adam(learning_rate),
                            ]))

print('..loading...')
load = Load('/home/xuehongyang/checkpoints_open/snapshot_18')
predictor = PredictDataStream(data_stream=data_stream_test,
                              output_tensor=result,
                              path='/home/xuehongyang/RESULT_MAIN',
                              before_training=True,
                              after_epoch=False,
                              after_training=False)
main_loop = MainLoop(model=Model(cost),
                     data_stream=data_stream_train,
                     algorithm=algorithm,
                     extensions=[
                         Timing(),
                         FinishAfter(after_n_epochs=1),
                         load,
                         predictor])

print('start prediction ...')
main_loop.run()
