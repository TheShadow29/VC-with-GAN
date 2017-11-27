import os
import json

import tensorflow as tf
import numpy as np

from analyzer import read_all, Tanhize
# load, configure_gpu_settings, restore_global_step
# from analyzer import read, Tanhize
from util.wrapper import save, validate_log_dirs
from importlib import import_module

args = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string(
    'logdir_root', None, 'root of log dir')
tf.app.flags.DEFINE_string(
    'logdir', None, 'log dir')
tf.app.flags.DEFINE_string(
    'restore_from', None, 'restore from dir (not from *.ckpt)')
tf.app.flags.DEFINE_string('gpu_cfg', None, 'GPU configuration')
tf.app.flags.DEFINE_integer('summary_freq', 1000, 'Update summary')
tf.app.flags.DEFINE_string(
    'ckpt', None, 'specify the ckpt in restore_from (if there are multiple ckpts)')  # TODO
tf.app.flags.DEFINE_string(
    'architecture', 'architecture-vawgan-vcc2016.json', 'network architecture')

tf.app.flags.DEFINE_string('model_module', 'model.vae', 'Model module')
tf.app.flags.DEFINE_string('model', None, 'Model: ConvVAE, VAWGAN')

tf.app.flags.DEFINE_string('trainer_module', 'trainer.vae', 'Trainer module')
tf.app.flags.DEFINE_string('trainer', None, 'Trainer: VAETrainer, VAWGANTrainer')

if args.model is None or args.trainer is None:
    raise ValueError(
        '\n  Both `model` and `trainer` should be assigned.' +
        '\n  Use `python main.py --help` to see applicable options.'
    )

# print(args.model_module)
model_module = import_module(args.model_module, package=None)
MODEL = getattr(model_module, args.model)

trainer_module = import_module(args.trainer_module, package=None)
TRAINER = getattr(trainer_module, args.trainer)
# print(args.model_module)

# MODEL = model.vae.VAWGAN
# TRAINER = trainer.vae.VAWGANTrainer


def main():
    """ NOTE: The input is rescaled to [-1, 1] """

    dirs = validate_log_dirs(args)
    tf.gfile.MakeDirs(dirs['logdir'])
    # dirs = dict()
    # dirs['logdir'] = '.'
    with open(args.architecture) as f:
        arch = json.load(f)

    with open(os.path.join(dirs['logdir'], args.architecture), 'w') as f:
        json.dump(arch, f, indent=4)

    normalizer = Tanhize(
        xmax=np.fromfile('./etc/xmax.npf'),
        xmin=np.fromfile('./etc/xmin.npf'),
    )

    image, label, text_emb = read_all(
        file_pattern=arch['training']['datadir'],
        batch_size=arch['training']['batch_size'],
        capacity=2048,
        min_after_dequeue=1024,
        normalizer=normalizer,
    )

    machine = MODEL(arch)

    if model_module == 'VAWGAN_S':
        loss = machine.loss(image, label, text_emb)
    else:
        loss = machine.loss(image, label)
    trainer = TRAINER(loss, arch, args, dirs)
    trainer.train(nIter=arch['training']['max_iter'], machine=machine)


if __name__ == '__main__':
    main()
