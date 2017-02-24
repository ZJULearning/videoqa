from __future__ import division, print_function

import os
import os.path
import shutil
import theano
import theano.tensor as T
import logging

from blocks.extensions import SimpleExtension, TrainingExtension
from blocks.utils import reraise_as
from blocks.serialization import (secure_dump, load, dump_and_add_to_dump,
                                                                    load_parameters)

logger = logging.getLogger(__name__)

LOADED_FROM = "loaded_from"
SAVED_TO = "saved_to"

from blocks.extensions.saveload import Checkpoint

class saveSnapshot(SimpleExtension):
    def __init__(self, path, parameters=None, save_separately=None,
                 save_main_loop=True, use_cpickle=False, **kwargs):
        self.epoch = 0
        kwargs.setdefault("after_training", True)
        super(saveSnapshot, self).__init__(**kwargs)
        self.path = path
        self.parameters = parameters
        self.save_separately = save_separately
        self.save_main_loop = save_main_loop
        self.use_cpickle = use_cpickle

    def do(self, callback_name, *args):
        """Pickle the main loop object to the disk.
        If `*args` contain an argument from user, it is treated as
        saving path to be used instead of the one given at the
        construction stage.
        """
        logger.info("Snapshoting has started")
        _, from_user = self.parse_args(callback_name, args)
        try:
            path = self.path + '_%d' % self.epoch
            if from_user:
                path, = from_user
            to_add = None
            if self.save_separately:
                to_add = {attr: getattr(self.main_loop, attr) for attr in
                          self.save_separately}
            if self.parameters is None:
                if hasattr(self.main_loop, 'model'):
                    self.parameters = self.main_loop.model.parameters
            object_ = None
            if self.save_main_loop:
                object_ = self.main_loop
            secure_dump(object_, path,
                        dump_function=dump_and_add_to_dump,
                        parameters=self.parameters,
                        to_add=to_add,
                        use_cpickle=self.use_cpickle)
            self.epoch += 1
        except Exception:
            path = None
            raise
        finally:
            already_saved_to = self.main_loop.log.current_row.get(SAVED_TO, ())
            self.main_loop.log.current_row[SAVED_TO] = (already_saved_to +
                                                        (path,))
            logger.info("Snapshoting has finished")

