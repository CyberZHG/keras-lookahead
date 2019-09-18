from .backend import keras, TF_KERAS
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

    @property
    def learning_rate(self):
        return self.optimizer.learning_rate

    @learning_rate.setter
    def learning_rate(self, learning_rate):
        self.optimizer.learning_rate = learning_rate

    @property
    def iterations(self):
        return self.optimizer.iterations

    def get_updates(self, loss, params):
        sync_cond = K.equal((self.iterations + 1) // self.sync_period * self.sync_period, (self.iterations + 1))
        if TF_KERAS:
            slow_params = [K.variable(K.get_value(p), name='sp_{}'.format(i)) for i, p in enumerate(params)]
            self.updates = self.optimizer.get_updates(loss, params)
            slow_updates = []
            for p, sp in zip(params, slow_params):
                sp_t = sp + self.slow_step * (p - sp)
                slow_updates.append(K.update(sp, K.switch(
                    sync_cond,
                    sp_t,
                    sp,
                )))
                slow_updates.append(K.update_add(p, K.switch(
                    sync_cond,
                    sp_t - p,
                    K.zeros_like(p),
                )))
        else:
            slow_params = {p.name: K.variable(K.get_value(p), name='sp_{}'.format(i)) for i, p in enumerate(params)}
            update_names = ['update', 'update_add', 'update_sub']
            original_updates = [getattr(K, name) for name in update_names]
            setattr(K, 'update', lambda x, new_x: ('update', x, new_x))
            setattr(K, 'update_add', lambda x, new_x: ('update_add', x, new_x))
            setattr(K, 'update_sub', lambda x, new_x: ('update_sub', x, new_x))
            self.updates = self.optimizer.get_updates(loss, params)
            for name, original_update in zip(update_names, original_updates):
                setattr(K, name, original_update)
            slow_updates = []
            for i, update in enumerate(self.updates):
                if isinstance(update, tuple):
                    name, x, new_x, adjusted = update + (update[-1],)
                    update_func = getattr(K, name)
                    if name == 'update_add':
                        adjusted = x + new_x
                    if name == 'update_sub':
                        adjusted = x - new_x
                    if x.name not in slow_params:
                        self.updates[i] = update_func(x, new_x)
                    else:
                        slow_param = slow_params[x.name]
                        slow_param_t = slow_param + self.slow_step * (adjusted - slow_param)
                        slow_updates.append(K.update(slow_param, K.switch(
                            sync_cond,
                            slow_param_t,
                            slow_param,
                        )))
                        self.updates[i] = K.update(x, K.switch(
                            sync_cond,
                            slow_param_t,
                            adjusted,
                        ))
            slow_params = list(slow_params.values())
        self.updates += slow_updates
        self.weights = self.optimizer.weights + slow_params
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
