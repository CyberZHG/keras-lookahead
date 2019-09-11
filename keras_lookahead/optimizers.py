from .backend import keras
from .backend import backend as K

__all__ = ['Lookahead']


class Lookahead(keras.optimizers.Optimizer):
    """The lookahead mechanism for optimizers.

    Default parameters follow those provided in the original paper.

    # Arguments
        optimizer: An existed optimizer.
        sync_period: int > 0. The synchronization period.
        slow_step: float, 0 < alpha < 1. The step size of slow weights.

    # References
        - [Lookahead Optimizer: k steps forward, 1 step back]
          (https://arxiv.org/pdf/1907.08610v1.pdf)
    """

    def __init__(self, optimizer, sync_period=5, slow_step=0.5, **kwargs):
        super(Lookahead, self).__init__(**kwargs)
        self.optimizer = keras.optimizers.get(optimizer)
        with K.name_scope(self.__class__.__name__):
            self.sync_period = K.variable(sync_period, dtype='int64', name='sync_period')
            self.slow_step = K.variable(slow_step, name='slow_step')

    @property
    def lr(self):
        return self.optimizer.lr

    @lr.setter
    def lr(self, lr):
        self.optimizer.lr = lr

    def get_updates(self, loss, params):
        slow_params = {p.name: K.variable(K.get_value(p), name='sp_{}'.format(i)) for i, p in enumerate(params)}
        sync_cond = K.equal((self.optimizer.iterations + 1) % self.sync_period, 0)
        original_update = getattr(K, 'update')
        setattr(K, 'update', lambda x, new_x: (x, new_x))
        self.updates = self.optimizer.get_updates(loss, params)
        setattr(K, 'update', original_update)
        slow_updates = []
        for i, update in enumerate(self.updates):
            if isinstance(update, tuple):
                if update[0].name not in slow_params:
                    self.updates[i] = K.update(update[0], update[1])
                else:
                    slow_param = slow_params[update[0].name]
                    slow_param_t = slow_param + self.slow_step * (update[1] - slow_param)
                    slow_updates.append(K.update(slow_param, K.switch(
                        sync_cond,
                        slow_param_t,
                        slow_param,
                    )))
                    self.updates[i] = K.update(update[0], K.switch(
                        sync_cond,
                        slow_param_t,
                        update[1],
                    ))
        self.updates += slow_updates
        return self.updates

    def get_config(self):
        config = {
            'optimizer': keras.optimizers.serialize(self.optimizer),
            'sync_period': int(K.get_value(self.sync_period)),
            'slow_step': float(K.get_value(self.slow_step)),
        }
        base_config = super(Lookahead, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        optimizer = keras.optimizers.deserialize(config.pop('optimizer'))
        return cls(optimizer, **config)
