import torch
import tensorflow as tf
import os
import logging
import numpy as np
import torch.distributed as dist

from models import utils as mutils
from models.ema import ExponentialMovingAverage
import losses
import likelihood, sampling

def restore_checkpoint(config, ckpt_dir, state, device):
  if not tf.io.gfile.exists(ckpt_dir):
    tf.io.gfile.makedirs(os.path.dirname(ckpt_dir))
    logging.warning(f"No checkpoint found at {ckpt_dir}. "
                    f"Returned the same state as input")
    return state
  else:
    logging.info(ckpt_dir + ' loaded ...')
    loaded_state = torch.load(ckpt_dir, map_location=device)
    state['optimizer'].load_state_dict(loaded_state['optimizer'])
    state['model'].load_state_dict(loaded_state['model'], strict=False)
    state['ema'].load_state_dict(loaded_state['ema'])
    state['step'] = loaded_state['step']
    return state


def save_checkpoint(config, ckpt_dir, state):
  saved_state = {
    'optimizer': state['optimizer'].state_dict(),
    'model': state['model'].state_dict(),
    'ema': state['ema'].state_dict(),
    'step': state['step']
  }
  torch.save(saved_state, ckpt_dir)

def create_name(prefix, name, ext):
  try:
    name = f'{prefix}_{int(name)}.{ext}'
  except:
    if len(name.split('.')) == 1:
      name = f'{prefix}_{name}.{ext}'
    else:
      name = name.split('/')[-1]
      name = f'{prefix}_{name.split(".")[0]}.{ext}'
  return name

def load_model(config, workdir, print_=True, sde=None):
  # Initialize model.
  score_model = mutils.create_model(config, sde)
  optimizer = losses.get_optimizer(config, score_model.parameters())
  ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
  state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)

  if print_:
    # print(score_model)
    model_parameters = filter(lambda p: p.requires_grad, score_model.parameters())
    model_params = sum([np.prod(p.size()) for p in model_parameters])
    total_num_params = sum([np.prod(p.size()) for p in score_model.parameters()])
    logging.info(f"model parameters: {model_params}")
    logging.info(f"total number of parameters: {total_num_params}")

  # Create checkpoints directory
  checkpoint_dir = os.path.join(workdir, "checkpoints")
  # Intermediate checkpoints to resume training after pre-emption in cloud environments
  checkpoint_meta_dir = os.path.join(workdir, "checkpoints-meta", "checkpoint.pth")
  tf.io.gfile.makedirs(checkpoint_dir)
  tf.io.gfile.makedirs(os.path.dirname(checkpoint_meta_dir))
  # Resume training when intermediate checkpoints are detected
  state = restore_checkpoint(config, checkpoint_meta_dir, state, config.device)

  return state, score_model, ema, checkpoint_dir, checkpoint_meta_dir

def get_loss_fns(config, sde, inverse_scaler, train=True):
  optimize_fn = losses.optimization_manager(config)
  train_step_fn = losses.get_step_fn(config, sde, train=train, optimize_fn=optimize_fn)
  nll_fn = likelihood.get_likelihood_fn(config, sde, inverse_scaler)
  nelbo_fn = likelihood.get_elbo_fn(config, sde, inverse_scaler=inverse_scaler)
  sampling_shape = (config.sampling.batch_size, config.data.num_channels, config.data.image_size, config.data.image_size)
  sampling_fn = sampling.get_sampling_fn(config, sde, sampling_shape, inverse_scaler, config.sampling.truncation_time)
  return train_step_fn, nll_fn, nelbo_fn, sampling_fn