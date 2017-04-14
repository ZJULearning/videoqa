from vatt import *
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

prob_dim = 3591
bs = 100
# max ans length + <EOS> + <Unknown>
max_ans_length = 22
joint_dim = 1024
model = rewatching(
    bs, max_ans_length,
    4096, 300,
    2048,
    1024,
    1024,
    1024,
    1024,
    2048,
    prob_dim,
    prob_dim)
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
cg = ComputationGraph(cost)

data_train = H5PYDataset('/home/xuehongyang/TGIF_open_161217.hdf5', which_sets=('train',),
                                                  subset=slice(0, 230689//bs*bs))
data_val = H5PYDataset('/home/xuehongyang/TGIF_open_161217.hdf5', which_sets=('validation',),
                                              subset=slice(0, 24696//bs*bs))

data_test = H5PYDataset('/home/xuehongyang/TGIF_open_161217.hdf5', which_sets=('test',),
                                               subset=slice(0, 32378//bs*bs))

data_stream_train = DataStream.default_stream(
        data_train,
        iteration_scheme=ShuffledScheme(data_train.num_examples, batch_size=bs))

data_stream_val = DataStream.default_stream(
        data_val,
        iteration_scheme=SequentialScheme(
                    data_val.num_examples, batch_size=bs))

data_stream_test = DataStream.default_stream(
        data_test,
        iteration_scheme=SequentialScheme(
                    data_test.num_examples, batch_size=bs))

monitor = TrainingDataMonitoring(
        variables=[cost], prefix='train', every_n_batches=500, after_epoch=True)


monitor_val = DataStreamMonitoring(
        variables=[cost, error], data_stream=data_stream_val, prefix='validation', after_epoch=True)

monitor_test = DataStreamMonitoring(
        variables=[error], data_stream=data_stream_test, prefix='test', after_epoch=True)

learning_rate = 0.00008
n_epochs=100
algorithm = GradientDescent(cost=cost,
                            parameters=cg.parameters,
                            on_unused_sources='ignore',
                            step_rule=CompositeRule([
                                StepClipping(10.),
                                Adam(learning_rate),
                            ]))

main_loop = MainLoop(model=Model(cost),
                    data_stream=data_stream_train,
                    algorithm=algorithm,
                    extensions=[
                        Timing(),
                        FinishAfter(after_n_epochs=n_epochs),
                        monitor,
                        monitor_val,
                        monitor_test,
                        saveSnapshot('/home/xuehongyang/checkpoints_vatt/snapshot',
                                     save_main_loop=False,
                                     after_epoch=True,
                                     save_separately=['log', 'model']),
                        ProgressBar(),
                        Printing(every_n_batches=500),
                        Plot('videoqa_open_videoattention', channels=[['train_cost']],
                             every_n_batches=500,
                             after_batch=True)])

print('starting...')

main_loop.run()
