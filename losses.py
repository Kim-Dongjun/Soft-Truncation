# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""All functions related to loss computation and optimization.
"""

import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pickle
from models import utils as mutils
from sde_lib import VESDE, VPSDE
import likelihood


def get_optimizer(config, params):
  """Returns a flax optimizer object based on `config`."""
  if config.optim.optimizer == 'Adam':
    optimizer = optim.Adam(params, lr=config.optim.lr, betas=(config.optim.beta1, 0.999), eps=config.optim.eps,
                           weight_decay=config.optim.weight_decay, amsgrad=config.optim.amsgrad)
  elif config.optim.optimizer == 'AdamW':
    optimizer = optim.AdamW(params, lr=config.optim.lr, betas=(config.optim.beta1, 0.99), eps=config.optim.eps,
                           weight_decay=config.optim.weight_decay)
  else:
    raise NotImplementedError(
      f'Optimizer {config.optim.optimizer} not supported yet!')

  return optimizer


def optimization_manager(config):
  """Returns an optimize_fn based on `config`."""

  def optimize_fn(optimizer, params, step, lr=config.optim.lr,
                  warmup=config.optim.warmup,
                  grad_clip=config.optim.grad_clip):
    """Optimizes with warmup and gradient clipping (disabled if negative)."""
    if warmup > 0:
      for g in optimizer.param_groups:
        g['lr'] = lr * np.minimum(step / warmup, 1.0)
    if grad_clip >= 0:
      torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip)
    optimizer.step()

  return optimize_fn


def get_sde_loss_fn(config, sde, train, variance='scoreflow'):
  """Create a loss function for training with arbirary SDEs.

  Args:
    sde: An `sde_lib.SDE` object that represents the forward SDE.
    train: `True` for training loss and `False` for evaluation loss.
    reduce_mean: If `True`, average the loss across data dimensions. Otherwise sum the loss across data dimensions.
    continuous: `True` indicates that the model is defined to take continuous time steps. Otherwise it requires
      ad-hoc interpolation to take continuous time steps.
    likelihood_weighting: If `True`, weight the mixture of score matching losses
      according to https://arxiv.org/abs/2101.09258; otherwise use the weighting recommended in our paper.
    eps: A `float` number. The smallest time step to sample from.

  Returns:
    A loss function.
  """
  reduce_op = torch.mean if config.training.reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)

  def approx_standard_normal_cdf(x):
    return 0.5 * (1.0 + torch.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * (x ** 3))))

  def discretized_gaussian_log_likelihood(x, means, log_scales):
    # Assumes data is integers [0, 255] rescaled to [-1, 1]
    assert x.shape == means.shape# == log_scales.shape
    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1. / 255.)
    cdf_plus = approx_standard_normal_cdf(plus_in)
    min_in = inv_stdv * (centered_x - 1. / 255.)
    cdf_min = approx_standard_normal_cdf(min_in)
    log_cdf_plus = torch.log(torch.max(cdf_plus, torch.tensor(1e-12, device=cdf_plus.device)))
    log_one_minus_cdf_min = torch.log(torch.max(1. - cdf_min, torch.tensor(1e-12, device=cdf_plus.device)))
    cdf_delta = cdf_plus - cdf_min
    log_probs = torch.where(
      x < -0.999, log_cdf_plus,
      torch.where(x > 0.999, log_one_minus_cdf_min,
               torch.log(torch.max(cdf_delta, torch.tensor(1e-12, device=cdf_delta.device)))))
    assert log_probs.shape == x.shape
    return log_probs

  def loss_fn(model, batch, importance_sampling, t_min=None):
    """Compute the loss function.

    Args:
      model: A score model.
      batch: A mini-batch of training data.

    Returns:
      loss: A scalar that represents the average loss value across the mini-batch.
    """
    if t_min == None:
      t_min = sde.get_t_min(config)
    t, Z = sde.get_diffusion_time(config, batch.shape[0], batch.device, t_min, importance_sampling=importance_sampling)

    score_fn = mutils.get_score_fn(config, sde, model, train=train, continuous=config.training.continuous)
    z = torch.randn_like(batch)

    mean, std = sde.marginal_prob(batch, t)
    perturbed_data = mean + std[:, None, None, None] * z
    score = score_fn(perturbed_data, t)

    if config.training.importance_sampling:
      losses = torch.square(score * std[:, None, None, None] + z)
      losses = 0.5 * Z * reduce_op(losses.reshape(losses.shape[0], -1), dim=-1)
    else:
      if config.training.likelihood_weighting:
        g2 = sde.sde(torch.zeros_like(batch), t)[1] ** 2
        losses = torch.square(score + z / std[:, None, None, None])
        losses = 0.5 * Z * reduce_op(losses.reshape(losses.shape[0], -1), dim=-1) * g2
      else:
        losses = torch.square(score * std[:, None, None, None] + z)
        losses = 0.5 * Z * reduce_op(losses.reshape(losses.shape[0], -1), dim=-1)

    if config.training.reconstruction_loss:
      eps_vec = torch.ones((batch.shape[0]), device=batch.device) * t_min
      mean, std = sde.marginal_prob(batch, eps_vec)
      z = torch.randn_like(batch)
      perturbed_data = mean + std[:, None, None, None] * z
      score = score_fn(perturbed_data, eps_vec)

      alpha, beta = sde.marginal_prob(torch.ones_like(batch), eps_vec)
      q_mean = perturbed_data / alpha + beta[:, None, None, None] ** 2 * score / alpha
      if variance == 'ddpm':
        q_std = beta
      elif variance == 'scoreflow':
        q_std = beta / torch.mean(alpha, axis=(1, 2, 3))

      if config.data.dequantization == 'lossless':
        decoder_nll = -discretized_gaussian_log_likelihood(
          batch, means=q_mean, log_scales=torch.log(q_std)[:, None, None, None])
        reconstruction_loss = (decoder_nll).sum(axis=(1, 2, 3))
      else:
        n_dim = np.prod(batch.shape[1:])
        p_entropy = n_dim / 2. * (np.log(2 * np.pi) + 2 * torch.log(std) + 1.)
        q_recon = n_dim / 2. * (np.log(2 * np.pi) + 2 * torch.log(q_std)) + 0.5 / (q_std ** 2) * torch.square(
          batch - q_mean).sum(axis=(1, 2, 3))
        assert q_recon.shape == p_entropy.shape == torch.Size([batch.shape[0]])
        reconstruction_loss = q_recon - p_entropy
        assert losses.shape == reconstruction_loss.shape

      if config.training.reduce_mean:
        reconstruction_loss = reconstruction_loss / np.prod(list(batch.shape[1:]))

      losses = losses + reconstruction_loss

    return losses

  return loss_fn


def get_smld_loss_fn(config, vesde, train):
  """Legacy code to reproduce previous results on SMLD(NCSN). Not recommended for new work."""
  assert isinstance(vesde, VESDE), "SMLD training only works for VESDEs."

  # Previous SMLD models assume descending sigmas
  smld_sigma_array = torch.flip(vesde.discrete_sigmas, dims=(0,))
  reduce_op = torch.mean if config.training.reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)

  def loss_fn(model, batch):
    model_fn = mutils.get_model_fn(model, train=train)
    labels = torch.randint(0, vesde.N, (batch.shape[0],), device=batch.device)
    sigmas = smld_sigma_array.to(batch.device)[labels]
    noise = torch.randn_like(batch) * sigmas[:, None, None, None]
    perturbed_data = noise + batch
    score = model_fn(perturbed_data, labels)
    target = -noise / (sigmas ** 2)[:, None, None, None]
    losses = torch.square(score - target)
    losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1) * sigmas ** 2
    loss = torch.mean(losses)
    return loss

  return loss_fn


def get_ddpm_loss_fn(config, vpsde, train):
  """Legacy code to reproduce previous results on DDPM. Not recommended for new work."""
  assert isinstance(vpsde, VPSDE), "DDPM training only works for VPSDEs."

  reduce_op = torch.mean if config.training.reduce_mean else lambda *args, **kwargs: 0.5 * torch.sum(*args, **kwargs)

  def loss_fn(model, batch):
    model_fn = mutils.get_model_fn(model, train=train)
    labels = torch.randint(0, vpsde.N, (batch.shape[0],), device=batch.device)
    sqrt_alphas_cumprod = vpsde.sqrt_alphas_cumprod.to(batch.device)
    sqrt_1m_alphas_cumprod = vpsde.sqrt_1m_alphas_cumprod.to(batch.device)
    noise = torch.randn_like(batch)
    perturbed_data = sqrt_alphas_cumprod[labels, None, None, None] * batch + \
                     sqrt_1m_alphas_cumprod[labels, None, None, None] * noise
    score = model_fn(perturbed_data, labels)
    losses = torch.square(score - noise)
    losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1)
    loss = torch.mean(losses)
    return loss

  return loss_fn


def get_step_fn(config, sde, train, optimize_fn=None):
  """Create a one-step training/evaluation function.

  Args:
    sde: An `sde_lib.SDE` object that represents the forward SDE.
    optimize_fn: An optimization function.
    reduce_mean: If `True`, average the loss across data dimensions. Otherwise sum the loss across data dimensions.
    continuous: `True` indicates that the model is defined to take continuous time steps.
    likelihood_weighting: If `True`, weight the mixture of score matching losses according to
      https://arxiv.org/abs/2101.09258; otherwise use the weighting recommended by our paper.

  Returns:
    A one-step function for training or evaluation.
  """
  if config.training.continuous:
    loss_fn = get_sde_loss_fn(config, sde, train)
  else:
    assert not config.training.likelihood_weighting, "Likelihood weighting is not supported for original SMLD/DDPM training."
    if isinstance(sde, VESDE):
      loss_fn = get_smld_loss_fn(config, sde, train)
    elif isinstance(sde, VPSDE):
      loss_fn = get_ddpm_loss_fn(config, sde, train)
    else:
      raise ValueError(f"Discrete training for {sde.__class__.__name__} is not recommended.")

  def step_fn_temp(state, batch, step):
    """Running one step of training or evaluation.

    This function will undergo `jax.lax.scan` so that multiple steps can be pmapped and jit-compiled together
    for faster execution.

    Args:
      state: A dictionary of training information, containing the score model, optimizer,
       EMA status, and number of optimization steps.
      batch: A mini-batch of training/evaluation data.

    Returns:
      loss: The average loss value of this state.
    """
    model = state['model']
    _ = loss_fn(model, batch, step)

    return torch.zeros(batch.shape[0])

  def step_fn(state, batch):
    """Running one step of training or evaluation.

    This function will undergo `jax.lax.scan` so that multiple steps can be pmapped and jit-compiled together
    for faster execution.

    Args:
      state: A dictionary of training information, containing the score model, optimizer,
       EMA status, and number of optimization steps.
      batch: A mini-batch of training/evaluation data.

    Returns:
      loss: The average loss value of this state.
    """
    model = state['model']
    optimizer = state['optimizer']

    if train:
      optimizer.zero_grad()
      batch_size = batch.shape[0]
      num_micro_batch = config.optim.num_micro_batch
      losses_ = torch.zeros(batch_size)
      t_min = sde.get_t_min(config)
      for k in range(num_micro_batch):
        losses = loss_fn(model, batch[batch_size // num_micro_batch * k: batch_size // num_micro_batch * (k + 1)], importance_sampling=config.training.importance_sampling, t_min=t_min)
        torch.mean(losses).backward(retain_graph=True)
        losses_[batch_size // num_micro_batch * k: batch_size // num_micro_batch * (k + 1)] = losses.cpu().detach()
      optimize_fn(optimizer, model.parameters(), step=state['step'])
      state['step'] += 1
      state['ema'].update(model.parameters())

    return losses_

  def step_fn_mixed(state, batch):
    model = state['model']
    optimizer = state['optimizer']

    if train:
      optimizer.zero_grad()
      batch_size = batch.shape[0]
      num_micro_batch = config.optim.num_micro_batch
      losses_ = torch.zeros(batch_size // 2)
      t_min = sde.get_t_min(config)
      for k in range(num_micro_batch):
        losses_is = loss_fn(model, batch[batch_size // num_micro_batch * k: batch_size // num_micro_batch * k + batch_size // (2 * num_micro_batch)],
                         importance_sampling=True, t_min=t_min)
        losses_ddpm = loss_fn(model, batch[batch_size // num_micro_batch * k + batch_size // (2 * num_micro_batch): batch_size // num_micro_batch * (k+1)],
                         importance_sampling=False, t_min=t_min)
        if config.training.balanced:
          losses = losses_is + config.training.ddpm_weight * torch.mean(losses_is / losses_ddpm).detach().item() * losses_ddpm
        else:
          losses = losses_is + config.training.ddpm_weight * losses_ddpm
        torch.mean(losses).backward(retain_graph=True)
        losses_[batch_size // num_micro_batch // 2 * k: batch_size // num_micro_batch // 2 * (k + 1)] = losses.cpu().detach()
      optimize_fn(optimizer, model.parameters(), step=state['step'])
      state['step'] += 1
      state['ema'].update(model.parameters())

    return losses_

  if config.training.mixed:
    return step_fn_mixed
  else:
    return step_fn

def get_div_fn(fn):
  """Create the divergence function of `fn` using the Hutchinson-Skilling trace estimator."""

  def div_fn(x, t, eps):
    with torch.enable_grad():
      #x.requires_grad_(True)
      fn_eps = torch.sum(fn(x, t) * eps)
      grad_fn_eps = torch.autograd.grad(fn_eps, x)[0]
    #x.requires_grad_(False)
    return torch.sum(grad_fn_eps * eps, dim=tuple(range(1, len(x.shape))))

  return div_fn