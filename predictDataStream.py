from __future__ import division, print_function

import os
import os.path
import shutil
import theano
import theano.tensor as T
import logging
import numpy as np
from blocks.extensions import SimpleExtension, TrainingExtension
from blocks.utils import reraise_as
from blocks.serialization import (secure_dump, load, dump_and_add_to_dump,
                                                                    load_parameters)
import sys
from blocks.graph import ComputationGraph


class PredictDataStream(SimpleExtension):
    """
    Predict for a given datastream.
    """
    def __init__(self, data_stream, output_tensor, path, **kwargs):
        self.data_stream = data_stream
        self.output_tensor = output_tensor
        self.prediction = None
        self.path = path
        kwargs.setdefault('before_training', True)
        super(PredictDataStream, self).__init__(**kwargs)
        
        cg1 = ComputationGraph(output_tensor)
        self.theano_function = cg1.get_theano_function(on_unused_input='ignore')
        self.iter = 0
    def do(self, which_callback, *args):
        prediction = []
        print('...predicting...')
        for batch in self.data_stream.get_epoch_iterator(as_dict=True):
            prediction.extend(self.theano_function(**batch))
        self.prediction = np.concatenate(prediction, axis=0)
        np.save('%s%d' % (self.path, self.iter), self.prediction, allow_pickle=False)
        print('...done saving')
        self.iter += 1
        sys.exit(0)
