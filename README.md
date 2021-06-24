# Keras Lookahead

[![Travis](https://travis-ci.com/CyberZHG/keras-lookahead.svg)](https://travis-ci.org/CyberZHG/keras-lookahead)
[![Coverage](https://coveralls.io/repos/github/CyberZHG/keras-lookahead/badge.svg?branch=master)](https://coveralls.io/github/CyberZHG/keras-lookahead)

**This repo is outdated and will no longer be maintained.**

Unofficial implementation of the [lookahead mechanism](https://arxiv.org/pdf/1907.08610v1.pdf) for optimizers.

## Install

```bash
pip install git+https://github.com/cyberzhg/keras-lookahead.git
```

## External Links

- [tensorflow/addons:LookAhead](https://github.com/tensorflow/addons/blob/master/tensorflow_addons/optimizers/lookahead.py)

## Usage

Arguments:

* `optimizer`: Original optimizer.
* `sync_period`: the `k` in the paper. The synchronization period.
* `slow_step`: the `α` in the paper. The step size of slow weights.

```python
from keras_lookahead import Lookahead

optimizer = Lookahead('adam', sync_period=5, slow_step=0.5)
```

Custom optimizers can also be used:

```python
from keras_radam import RAdam
from keras_lookahead import Lookahead

optimizer = Lookahead(RAdam())
```
