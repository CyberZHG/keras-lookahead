import os
import tempfile
from unittest import TestCase

import numpy as np

from keras_lookahead.backend import keras
from keras_lookahead import Lookahead


class TestLookahead(TestCase):

    @staticmethod
    def _init_data(data_size=1024, w=None):
        x = np.random.standard_normal((data_size, 5))
        if w is None:
            w = np.random.standard_normal((5, 1))
        y = np.dot(x, w) + np.random.standard_normal((data_size, 1)) * 1e-6
        return x, y, w

    @staticmethod
    def _init_model(optimizer, w=None):
        inputs = keras.layers.Input(shape=(5,))
        dense = keras.layers.Dense(1)
        outputs = dense(inputs)
        model = keras.models.Model(inputs, outputs)
        model.compile(optimizer, 'mse')
        if w is not None:
            dense.set_weights([w, np.zeros((1,))])
        return model

    def test_base(self):
        model = self._init_model(Lookahead('adam'))
        x, y, w = self._init_data(data_size=100000)
        model.fit(x, y, epochs=5)

        model_path = os.path.join(tempfile.gettempdir(), 'test_lookahead_%f.h5' % np.random.random())
        model.save(model_path)
        model: keras.models.Model = keras.models.load_model(model_path, custom_objects={'Lookahead': Lookahead})

        x, y, _ = self._init_data(data_size=100, w=w)
        predicted = model.predict(x)
        self.assertLess(np.max(np.abs(predicted - y)), 1e-3)

    def test_half(self):
        weight = np.random.standard_normal((5, 1))
        x, y, _ = self._init_data(data_size=320)

        model = self._init_model('adam', w=weight)
        model.fit(x, y, batch_size=32)
        original = model.get_weights()[0]

        model = self._init_model(Lookahead('adam', sync_period=10, slow_step=0.5), w=weight)
        model.fit(x, y, batch_size=32)
        step_back = model.get_weights()[0]

        half_step = (weight + original) * 0.5
        self.assertTrue(np.allclose(half_step, step_back, atol=1e-2))
