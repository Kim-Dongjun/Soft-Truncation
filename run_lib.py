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
"""Training and evaluation for score-based generative models. """
import os
import numpy as np
import tensorflow as tf
import logging
# Keep the import below for registering all model definitions
from models import ddpm, ncsnv2, ncsnpp
import sampling_lib
import datasets
import evaluation
import sde_lib
from absl import flags
import torch
import utils
import losses

FLAGS = flags.FLAGS


def train(config, workdir, assetdir):
  """Runs the training pipeline.

  Args:
    config: Configuration to use.
    workdir: Working directory for checkpoints and TF summaries. If this
      contains checkpoint training will be resumed from the latest checkpoint.
  """
  sample_dir = os.path.join(workdir, "samples")
  tf.io.gfile.makedirs(sample_dir)

  tb_dir = os.path.join(workdir, "tensorboard")
  tf.io.gfile.makedirs(tb_dir)

  # Setup SDEs
  sde = sde_lib.get_sde(config, None)

  # Initialize model.
  state, score_model, ema, checkpoint_dir, checkpoint_meta_dir = utils.load_model(config, workdir, sde=sde)
  initial_step = int(state['step'])

  # Build data iterators
  logging.info(f'loading {config.data.dataset}...')
  train_ds, eval_ds = datasets.get_dataset(config)
  train_iter = iter(train_ds)  # pytype: disable=wrong-arg-types
  # Create data normalizer and its inverse
  scaler = datasets.get_data_scaler(config)
  inverse_scaler = datasets.get_data_inverse_scaler(config)

  # Build one-step loss functions
  train_step_fn, nll_fn, nelbo_fn, sampling_fn = utils.get_loss_fns(config, sde, inverse_scaler)

  # In case there are multiple hosts (e.g., TPU pods), only log to host 0
  logging.info("Starting training loop at step %d." % (initial_step,))
  for step in range(initial_step, config.training.n_iters + 1):
    # Convert data to JAX arrays and normalize them. Use ._numpy() to avoid copy.
    batch, train_iter = datasets.get_batch(config, train_iter, train_ds)
    if config.data.dequantization == 'uniform':
      batch = (255. * batch + torch.rand_like(batch)) / 256.
    batch = scaler(batch)
    # Execute one training step
    losses_ = train_step_fn(state, batch)
    if step % config.training.log_freq == 0:
      logging.info("step: %d, training loss mean: %.5e, training loss std: %.5e" % (step, torch.mean(losses_).item(), torch.std(losses_).item()))

    # Save a temporary checkpoint to resume training after pre-emption periodically
    if step != 0 and step % config.training.snapshot_freq_for_preemption == 0:
      utils.save_checkpoint(config, checkpoint_meta_dir, state)

    # Save a checkpoint periodically and generate samples if needed
    if step != 0 and step % config.training.snapshot_freq == 0 or step == config.training.n_iters:
      # Save the checkpoint.
      save_step = step // config.training.snapshot_freq
      utils.save_checkpoint(config, os.path.join(checkpoint_dir, f'checkpoint_{save_step}.pth'), state)

    if step != 0 and step % config.training.snapshot_freq == 0:
      if config.eval.enable_bpd:
        torch.cuda.empty_cache()
        ema.store(score_model.parameters())
        ema.copy_to(score_model.parameters())
        evaluation.compute_bpd(config, eval_ds, scaler, inverse_scaler, nelbo_fn, nll_fn, score_model, state, step=step)
        ema.restore(score_model.parameters())
        torch.cuda.empty_cache()

    if step != 0 and step % config.training.snapshot_freq == 0 or step == config.training.n_iters or config.training.whatever_sampling:
      # Generate and save samples
      if config.training.snapshot_sampling:
        logging.info('sampling start ...')
        torch.cuda.empty_cache()
        ema.store(score_model.parameters())
        ema.copy_to(score_model.parameters())
        for _ in range((config.eval.num_samples - 1) // config.sampling.batch_size + 1):
          sampling_lib.get_samples(config, score_model, state, sampling_fn, step, np.random.randint(1000000), sample_dir)
        ema.restore(score_model.parameters())
        torch.cuda.empty_cache()
        logging.info('sampling end ... computing FID ...')
        evaluation.compute_fid_and_is(config, score_model, state, sampling_fn, step, sample_dir, assetdir, config.eval.num_samples)
        torch.cuda.empty_cache()

def evaluate(config,
             workdir,
             assetdir,
             eval_folder="eval"):
  """Evaluate trained models.

  Args:
    config: Configuration to use.
    workdir: Working directory for checkpoints.
    eval_folder: The subfolder for storing evaluation results. Default to
      "eval".
  """
  eval_dir = os.path.join(workdir, eval_folder)
  tf.io.gfile.makedirs(eval_dir)

  # Create data normalizer and its inverse
  scaler = datasets.get_data_scaler(config)
  inverse_scaler = datasets.get_data_inverse_scaler(config)

  # Setup SDEs
  sde = sde_lib.get_sde(config, None)

  # Initialize model.
  state, score_model, ema, checkpoint_dir, checkpoint_meta_dir = utils.load_model(config, workdir, sde=sde)
  logging.info(f'score model step: {int(state["step"])}')
  ema.copy_to(score_model.parameters())

  # Build one-step loss functions
  _, nll_fn, nelbo_fn, sampling_fn = utils.get_loss_fns(config, sde, inverse_scaler)

  # Create data loaders for likelihood evaluation. Only evaluate on uniformly dequantized data
  train_ds, eval_ds = datasets.get_dataset(config)

  if config.eval.enable_bpd:
    evaluation.compute_bpd(config, eval_ds, scaler, inverse_scaler, nelbo_fn, nll_fn, score_model, step=int(state['step']), eval=True)

  if config.eval.enable_sampling:
    sample_dir = os.path.join(workdir, "eval")
    step = int(state['step'])
    logging.info('sampling start ...')
    torch.cuda.empty_cache()
    ema.copy_to(score_model.parameters())
    if config.sampling.sample_more:
      for _ in range((config.eval.num_samples - 1) // config.sampling.batch_size + 1):
        sampling_lib.get_samples(config, score_model, state, sampling_fn, step, np.random.randint(1000000), sample_dir)
    ema.restore(score_model.parameters())
    torch.cuda.empty_cache()
    logging.info('sampling end ... computing FID ...')
    evaluation.compute_fid_and_is(config, score_model, state, sampling_fn, step, sample_dir, assetdir, config.eval.num_samples)