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

# pylint: skip-file
# pytype: skip-file
"""Various sampling methods."""

import torch
import torch.nn.functional as F
import numpy as np
from scipy import integrate
from models import utils as mutils
import logging

def get_div_fn(fn):
  """Create the divergence function of `fn` using the Hutchinson-Skilling trace estimator."""

  def div_fn(x, t, eps):
    with torch.enable_grad():
      x.requires_grad_(True)
      fn_eps = torch.sum(fn(x, t) * eps)
      grad_fn_eps = torch.autograd.grad(fn_eps, x)[0]
    x.requires_grad_(False)
    return torch.sum(grad_fn_eps * eps, dim=tuple(range(1, len(x.shape))))

  return div_fn



def get_likelihood_fn(config, sde, inverse_scaler, hutchinson_type='Rademacher',
                      rtol=1e-5, atol=1e-5, method='RK45'):
  """Create a function to compute the unbiased log-likelihood estimate of a given data point.

  Args:
    sde: A `sde_lib.SDE` object that represents the forward SDE.
    inverse_scaler: The inverse data normalizer.
    hutchinson_type: "Rademacher" or "Gaussian". The type of noise for Hutchinson-Skilling trace estimator.
    rtol: A `float` number. The relative tolerance level of the black-box ODE solver.
    atol: A `float` number. The absolute tolerance level of the black-box ODE solver.
    method: A `str`. The algorithm for the black-box ODE solver.
      See documentation for `scipy.integrate.solve_ivp`.
    eps: A `float` number. The probability flow ODE is integrated to `eps` for numerical stability.

  Returns:
    A function that a batch of data points and returns the log-likelihoods in bits/dim,
      the latent code, and the number of function evaluations cost by computation.
  """

  def drift_fn(model, x, t):
    """The drift function of the reverse-time SDE."""
    score_fn = mutils.get_score_fn(config, sde, model, train=False, continuous=True)
    # Probability flow ODE is a special case of Reverse SDE
    rsde = sde.reverse(score_fn, probability_flow=config.eval.probability_flow, lambda_=config.eval.lambda_)
    return rsde.sde(x, t)[0]

  def div_fn(model, x, t, noise):
    return get_div_fn(lambda xx, tt: drift_fn(model, xx, tt))(x, t, noise)

  def likelihood_fn(model, data, logdet=0., eps=1e-5, mode='correct'):
    """Compute an unbiased estimate to the log-likelihood in bits/dim.

    Args:
      model: A score model.
      data: A PyTorch tensor.

    Returns:
      bpd: A PyTorch tensor of shape [batch size]. The log-likelihoods on `data` in bits/dim.
      z: A PyTorch tensor of the same shape as `data`. The latent representation of `data` under the
        probability flow ODE.
      nfe: An integer. The number of function evaluations used for running the black-box ODE solver.
    """
    with torch.no_grad():
      score_fn = mutils.get_score_fn(config, sde, model, train=False, continuous=True)
      shape = data.shape
      if hutchinson_type == 'Gaussian':
        epsilon = torch.randn_like(data)
      elif hutchinson_type == 'Rademacher':
        epsilon = torch.randint_like(data, low=0, high=2).float() * 2 - 1.
      else:
        raise NotImplementedError(f"Hutchinson type {hutchinson_type} unknown.")

      def ode_func(t, x):
        sample = mutils.from_flattened_numpy(x[:-shape[0]], shape).to(data.device).type(torch.float32)
        vec_t = torch.ones(sample.shape[0], device=sample.device) * t
        drift = mutils.to_flattened_numpy(drift_fn(model, sample, vec_t))
        logp_grad = mutils.to_flattened_numpy(div_fn(model, sample, vec_t, epsilon))
        return np.concatenate([drift, logp_grad], axis=0)

      if mode == 'correct':
        z = torch.randn_like(data)
        mean, std = sde.marginal_prob(data, torch.ones(data.shape[0], device=data.device) * eps)
        perturbed_data = mean + std[:, None, None, None] * z
        init = np.concatenate([mutils.to_flattened_numpy(perturbed_data), np.zeros((shape[0],))], axis=0)
      elif mode == 'wrong':
        init = np.concatenate([mutils.to_flattened_numpy(data), np.zeros((shape[0],))], axis=0)
      else:
        raise NotImplementedError

      solution = integrate.solve_ivp(ode_func, (eps, sde.T), init, rtol=rtol, atol=atol, method=method)
      nfe = solution.nfev
      zp = solution.y[:, -1]
      z = mutils.from_flattened_numpy(zp[:-shape[0]], shape).to(data.device).type(torch.float32)
      delta_logp = mutils.from_flattened_numpy(zp[-shape[0]:], (shape[0],)).to(data.device).type(torch.float32)
      prior_logp = sde.prior_logp(z)
      #print("score bpd: ", - torch.mean(prior_logp + delta_logp) / np.prod(list(data.shape[1:])) / np.log(2) + 7. - inverse_scaler(-1.))

      if mode == 'correct':
        residual_fn = get_likelihood_residual_fn(config, sde, score_fn, variance='scoreflow')
        residual_nll = residual_fn(data, eps)
        #print("residual bpd: ", torch.mean(residual_nll) / np.prod(list(data.shape[1:])) / np.log(2))
        delta_logp = delta_logp - residual_nll

      #print("logdet bpd: ", - torch.mean(logdet) / np.prod(list(batch.shape[1:])) / np.log(2))
      bpd = -(prior_logp + delta_logp + logdet) / np.log(2)
      N = np.prod(shape[1:])
      bpd = bpd / N
      # A hack to convert log-likelihoods to bits/dim
      offset = 7. - inverse_scaler(-1.)
      bpd = bpd + offset
      return bpd, z, nfe

  return likelihood_fn

def get_elbo_fn(config, sde, inverse_scaler=None, hutchinson_type='Rademacher'):
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

  @torch.enable_grad()
  def loss_fn(model, batch, logdet=0., eps=1e-5):
    """Compute the loss function.

    Args:
      model: A score model.
      batch: A mini-batch of training data.

    Returns:
      loss: A scalar that represents the average loss value across the mini-batch.
    """

    score_fn = mutils.get_score_fn(config, sde, model, train=False, continuous=True)
    time, Z = sde.get_diffusion_time(config, batch.shape[0], batch.device, eps, importance_sampling=True)
    #time = torch.linspace(sde.eps, 1., batch.shape[0], device=config.device)
    #time = torch.fmod(np.random.rand() + time, 1.) * (sde.T - sde.eps) + sde.eps
    if config.training.sde.lower() == 'reciprocal_vesde':
      qt = 1. / (1. / eps - 1. / sde.T)
      #qt = 1. / eps - 1. / sde.T
    else:
      qt = 1 / (sde.T - eps)
    z = torch.randn_like(batch)
    mean, std = sde.marginal_prob(batch, time)
    perturbed_data = mean + std[:, None, None, None] * z
    perturbed_data = perturbed_data.requires_grad_()
    score = score_fn(perturbed_data, time)
    f, g = sde.sde(perturbed_data, time)
    a = std[:, None, None, None] * score
    mu = (std[:, None, None, None] ** 2) * score - (std[:, None, None, None] ** 2) / (g[:, None, None, None] ** 2) * f

    if hutchinson_type == 'Gaussian':
      epsilon = torch.randn_like(batch)
    elif hutchinson_type == 'Rademacher':
      epsilon = torch.randint_like(batch, low=0, high=2).float() * 2 - 1.
    else:
      raise NotImplementedError(f"Hutchinson type {hutchinson_type} unknown.")

    Mu = - (
      torch.autograd.grad(mu, perturbed_data, epsilon, create_graph=False)[0] * epsilon
    ).reshape(batch.size(0), -1).sum(1, keepdim=False) * Z / qt

    Nu = - (a ** 2).reshape(batch.size(0), -1).sum(1, keepdim=False) * Z / 2 / qt

    lp_t = torch.ones_like(time) * sde.T
    lp_z = torch.randn_like(batch)
    lp_mean, lp_std = sde.marginal_prob(batch, lp_t)
    lp_perturbed_data = lp_mean + lp_std[:, None, None, None] * lp_z
    lp = sde.prior_logp(lp_perturbed_data)

    elbos = lp + (Mu + Nu) * (1. if config.training.sde.lower() != 'reciprocal_vesde' else 2. * eps * np.log(sde.sigma_max / sde.sigma_min))

    residual_fn = get_likelihood_residual_fn(config, sde, score_fn, variance='scoreflow')
    return - (elbos + logdet) / np.prod(list(batch.shape[1:])) / np.log(2) + 7. - inverse_scaler(-1.), residual_fn(batch, eps) / np.prod(list(batch.shape[1:])) / np.log(2)
    #return None, residual_fn(batch, eps) / np.prod(list(batch.shape[1:])) / np.log(2)

  return loss_fn

def get_likelihood_residual_fn(config, sde, score_fn, variance='ddpm'):
  """Create a function to compute the unbiased log-likelihood bound of a given data point.
  """

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

  def likelihood_residual_fn_lossless_compression(batch, eps=None):
    """Compute an unbiased estimate to the log-likelihood in bits/dim.

        Args:
          prng: An array of random states. The list dimension equals the number of devices.
          pstate: Replicated training state for running on multiple devices.
          data: A JAX array of shape [#devices, batch size, ...].

        Returns:
          bpd: A JAX array of shape [#devices, batch size]. The log-likelihoods on `data` in bits/dim.
          N: same as input
        """

    if eps == None:
      eps = sde.eps
    eps_vec = torch.ones((batch.shape[0]), device=batch.device) * eps
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

    if not config.data.centered:
      batch = 2. * batch - 1.
      q_mean = 2. * q_mean - 1.
      q_std = 2. * q_std

    decoder_nll = -discretized_gaussian_log_likelihood(
      batch, means=q_mean, log_scales=torch.log(q_std)[:, None, None, None])
    p_entropy = np.prod(batch.shape[1:]) / 2. * (np.log(2 * np.pi) + 2 * torch.log(std) + 1.)
    residual = (decoder_nll).sum(axis=(1, 2, 3)) - p_entropy

    assert residual.shape == torch.Size([batch.shape[0]])
    return residual

  def likelihood_residual_fn(batch, eps=None):
    """Compute an unbiased estimate to the log-likelihood in bits/dim.

    Args:
      prng: An array of random states. The list dimension equals the number of devices.
      pstate: Replicated training state for running on multiple devices.
      data: A JAX array of shape [#devices, batch size, ...].

    Returns:
      bpd: A JAX array of shape [#devices, batch size]. The log-likelihoods on `data` in bits/dim.
      N: same as input
    """
    if eps == None:
      eps = sde.eps
    eps_vec = torch.ones((batch.shape[0]), device=batch.device) * eps
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

    n_dim = np.prod(batch.shape[1:])
    p_entropy = n_dim / 2. * (np.log(2 * np.pi) + 2 * torch.log(std) + 1.)
    q_recon = n_dim / 2. * (np.log(2 * np.pi) + 2 * torch.log(q_std)) + 0.5 / (q_std ** 2) * torch.square(batch - q_mean).sum(axis=(1, 2, 3))
    residual = q_recon - p_entropy
    assert q_recon.shape == p_entropy.shape == torch.Size([batch.shape[0]])
    return residual

  if config.data.dequantization == 'lossless':
    return likelihood_residual_fn_lossless_compression
  else:
    return likelihood_residual_fn