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

# Modified at 2021 by anonymous authors of "Score Matching Model for Unbounded Data Score"
# submitted on NeurIPS 2021 conference.

# pylint: skip-file
"""Training and evaluation for score-based generative models. """

import gc
import io
import os
import time

import numpy as np
import tensorflow as tf
import tensorflow_gan as tfgan
import logging
# Keep the import below for registering all model definitions
from models import ddpm, ncsnv2, ncsnpp
import losses
import sampling
from models import utils as mutils
from models.ema import ExponentialMovingAverage
import datasets
from datasets import get_batch
import evaluation
import sampling_lib
import save
import likelihood
import sde_lib
from absl import flags
import torch
from torch.utils import tensorboard
from torchvision.utils import make_grid, save_image
from utils import save_checkpoint, restore_checkpoint

FLAGS = flags.FLAGS


def train(config, workdir):
  """Runs the training pipeline.

  Args:
    config: Configuration to use.
    workdir: Working directory for checkpoints and TF summaries. If this
      contains checkpoint training will be resumed from the latest checkpoint.
  """

  # Create directories for experimental logs
  sample_dir = os.path.join(workdir, "samples")
  tf.io.gfile.makedirs(sample_dir)

  tb_dir = os.path.join(workdir, "tensorboard")
  tf.io.gfile.makedirs(tb_dir)
  writer = tensorboard.SummaryWriter(tb_dir)

  # Setup SDEs
  if config.training.sde.lower() == 've-sde':
    sde = sde_lib.VESDE(transform_type=config.add.transform, sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales)
    sampling_eps = 1e-5
  elif config.training.sde.lower() == 'rve-sde':
    sde = sde_lib.RVESDE(transform_type=config.add.transform, eta=config.add.eta,
                                   sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales)
    sampling_eps = 1e-5
  else:
    raise NotImplementedError(f"SDE {config.training.sde} unknown.")

  # Initialize model.
  score_model = mutils.create_model(config, sde)
  ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
  optimizer = losses.get_optimizer(config, score_model.parameters())
  state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)

  # Create checkpoints directory
  checkpoint_dir = os.path.join(workdir, "checkpoints")
  # Intermediate checkpoints to resume training after pre-emption in cloud environments
  checkpoint_meta_dir = os.path.join(workdir, "checkpoints-meta", "checkpoint.pth")
  tf.io.gfile.makedirs(checkpoint_dir)
  tf.io.gfile.makedirs(os.path.dirname(checkpoint_meta_dir))
  # Resume training when intermediate checkpoints are detected
  state = restore_checkpoint(checkpoint_meta_dir, state, config.device)
  initial_step = int(state['step'])

  # Build data iterators
  train_ds, eval_ds, _ = datasets.get_dataset(config, uniform_dequantization=config.data.uniform_dequantization,
                                              batch_size=config.training.batch_size)
  _, ds_bpd, _ = datasets.get_dataset(config, uniform_dequantization=config.data.uniform_dequantization,
                                      batch_size=config.eval.batch_size, evaluation=True)
  train_iter = iter(train_ds)  # pytype: disable=wrong-arg-types
  eval_iter = iter(eval_ds)  # pytype: disable=wrong-arg-types

  # Create data normalizer and its inverse
  scaler = datasets.get_data_scaler(config)
  inverse_scaler = datasets.get_data_inverse_scaler(config)

  # Build one-step training and evaluation functions
  optimize_fn = losses.optimization_manager(config)
  continuous = config.training.continuous
  reduce_mean = config.training.reduce_mean
  likelihood_weighting = config.training.likelihood_weighting
  train_step_fn = losses.get_step_fn(config, sde, train=True, optimize_fn=optimize_fn,
                                     reduce_mean=reduce_mean, continuous=continuous,
                                     likelihood_weighting=likelihood_weighting)
  eval_step_fn = losses.get_step_fn(config, sde, train=False, optimize_fn=optimize_fn,
                                    reduce_mean=reduce_mean, continuous=continuous,
                                    likelihood_weighting=likelihood_weighting)
  likelihood_fn = likelihood.get_likelihood_fn(sde, inverse_scaler)

  # Building sampling functions
  if config.training.snapshot_sampling:
    sampling_shape = (config.eval.batch_size, config.data.num_channels,
                      config.data.image_size, config.data.image_size)
    sampling_fn = sampling.get_sampling_fn(config, sde, sampling_shape, inverse_scaler, sampling_eps)

  num_train_steps = config.training.n_iters

  # In case there are multiple hosts (e.g., TPU pods), only log to host 0
  logging.info("Starting training loop at step %d." % (initial_step,))
  for step in range(initial_step, num_train_steps + 1):
    try:
      batch = get_batch(config, next(train_iter), scaler)
    except:
      logging.info("New Epoch Start")
      train_iter = iter(train_ds)
      batch = get_batch(config, next(train_iter), scaler)
    # Execute one training step
    loss = train_step_fn(state, batch)

    if step % config.training.log_freq == 0:
      logging.info("step: %d, training_loss: %.5e" % (step, loss.item()))
      writer.add_scalar("training_loss", loss, step)

    # Save a temporary checkpoint to resume training after pre-emption periodically
    if step != 0 and step % config.training.snapshot_freq_for_preemption == 0:
      save_checkpoint(checkpoint_meta_dir, state)

    if step != 0 and step % config.training.snapshot_freq_for_preemption == 0:
      bpd_iter = iter(ds_bpd)  # pytype: disable=wrong-arg-types
      bpds = ['{}-th step'.format(step)]
      for batch_id in range(1):
        eval_batch = get_batch(config, next(bpd_iter), scaler, train=False)
        with torch.no_grad():
          bpd = likelihood_fn(score_model, eval_batch)[0]
        bpd = bpd.detach().cpu().numpy().reshape(-1)
        bpds.extend(bpd)
        print("mean bpd: %.5e" % np.mean(bpd))
        logging.info("step: %d, mean bpd: %.5e" % (step, np.mean(bpds[1:])))
        # Save bits/dim to disk or Google Cloud Storage
      print("step: %d, mean bpd: %.5e" % (step, np.mean(bpds[1:])))

    # Report the loss on an evaluation dataset periodically
    if step % config.training.eval_freq == 0:
      eval_batch = get_batch(config, next(eval_iter), scaler, train=False)
      eval_loss = eval_step_fn(state, eval_batch)
      logging.info("step: %d, eval_loss: %.5e" % (step, eval_loss.item()))
      writer.add_scalar("eval_loss", eval_loss.item(), step)

    # Save a checkpoint periodically and generate samples if needed
    if step != 0 and step % config.training.snapshot_freq == 0 or step == num_train_steps:
      # Save the checkpoint.
      save_step = step // config.training.snapshot_freq
      save_checkpoint(os.path.join(checkpoint_dir, f'checkpoint_{save_step}.pth'), state)

    if step != 0 and step % config.training.snapshot_freq == 0 or step == num_train_steps:
      # Generate and save samples
      if config.training.snapshot_sampling:
        print("sampling start ...")
        ema.store(score_model.parameters())
        ema.copy_to(score_model.parameters())
        sample, n = sampling_fn(score_model)
        ema.restore(score_model.parameters())
        this_sample_dir = os.path.join(sample_dir, "iter_{}".format(step))
        tf.io.gfile.makedirs(this_sample_dir)
        nrow = int(np.sqrt(sample.shape[0]))
        image_grid = make_grid(sample, nrow, padding=2)
        sample = np.clip(sample.permute(0, 2, 3, 1).cpu().numpy() * 255, 0, 255).astype(np.uint8)
        with tf.io.gfile.GFile(
            os.path.join(this_sample_dir, "sample.npz"), "wb") as fout:
          np.save(fout, sample)
        with tf.io.gfile.GFile(
            os.path.join(this_sample_dir, "sample.png"), "wb") as fout:
          save_image(image_grid, fout)
        print("sampling end ...")

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
  # Create directory to eval_folder
  eval_dir = os.path.join(workdir, eval_folder)
  tf.io.gfile.makedirs(eval_dir)

  # Create data normalizer and its inverse
  scaler = datasets.get_data_scaler(config)
  inverse_scaler = datasets.get_data_inverse_scaler(config)

  num_scales = config.model.num_scales

  # Setup SDEs
  if config.training.sde.lower() == 'vesde':
    sde = sde_lib.VESDE(transform_type=config.add.transform, sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=num_scales)
    sampling_eps = 1e-5
  elif config.training.sde.lower() == 'rve-sde':
    sde = sde_lib.RVESDE(transform_type=config.add.transform, eta=config.add.eta, sigma_min=config.model.sigma_min,
                                   sigma_max=config.model.sigma_max, N=num_scales)
    sampling_eps = 1e-5
  else:
    raise NotImplementedError(f"SDE {config.training.sde} unknown.")

  # Initialize model
  score_model = mutils.create_model(config, sde)
  optimizer = losses.get_optimizer(config, score_model.parameters())
  ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
  state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0)

  checkpoint_dir = os.path.join(workdir, "checkpoints-meta")

  # Create the one-step evaluation function when loss computation is enabled
  if config.eval.enable_loss:
    optimize_fn = losses.optimization_manager(config)
    continuous = config.training.continuous
    likelihood_weighting = config.training.likelihood_weighting

    reduce_mean = config.training.reduce_mean
    eval_step = losses.get_step_fn(config, sde, train=False, optimize_fn=optimize_fn,
                                   reduce_mean=reduce_mean,
                                   continuous=continuous,
                                   likelihood_weighting=likelihood_weighting)

  if config.eval.enable_bpd:
    train_ds_bpd, eval_ds_bpd, _ = datasets.get_dataset(config, batch_size=config.eval.batch_size,
                                                        uniform_dequantization=config.data.uniform_dequantization,
                                                        evaluation=True)

    if config.eval.bpd_dataset.lower() == 'train':
      ds_bpd = train_ds_bpd
      bpd_num_repeats = 1
    elif config.eval.bpd_dataset.lower() == 'test':
      # Go over the dataset 5 times when computing likelihood on the test dataset
      ds_bpd = eval_ds_bpd
      bpd_num_repeats = 5
    else:
      raise ValueError(f"No bpd dataset {config.eval.bpd_dataset} recognized.")

    # Build the likelihood computation function when likelihood is enabled
    if config.eval.enable_bpd:
      likelihood_fn = likelihood.get_likelihood_fn(sde, inverse_scaler)

  # Build the sampling function when sampling is enabled

  begin_ckpt = config.eval.begin_ckpt
  end_ckpt = config.eval.begin_ckpt
  logging.info("begin checkpoint: %d" % (begin_ckpt,))
  for ckpt in range(begin_ckpt, end_ckpt + 1):
    this_sample_dir = os.path.join(eval_dir, f"ckpt_{ckpt}")
    if config.eval.enable_sampling:
      sampling_shape = (config.eval.batch_size,
                        config.data.num_channels,
                        config.data.image_size, config.data.image_size)
      sampling_fn = sampling.get_sampling_fn(config, sde, sampling_shape, inverse_scaler, sampling_eps,
                                             sample_dir=this_sample_dir)
    # Wait if the target checkpoint doesn't exist yet
    waiting_message_printed = False
    ckpt_filename = os.path.join(checkpoint_dir, "checkpoint.pth")
    logging.info(f"ckpt_filename: {ckpt_filename}")
    while not tf.io.gfile.exists(ckpt_filename):
      if not waiting_message_printed:
        logging.warning("Waiting for the arrival of checkpoint_%d" % (ckpt,))
        waiting_message_printed = True
      time.sleep(60)
    ckpt_path = os.path.join(checkpoint_dir, f'checkpoint.pth')
    state = restore_checkpoint(ckpt_path, state, device=config.device)
    ema.copy_to(score_model.parameters())

    # Compute log-likelihoods (bits/dim) if enabled
    if config.eval.enable_bpd:
      bpds = []
      for repeat in range(bpd_num_repeats):
        bpd_iter = iter(ds_bpd)  # pytype: disable=wrong-arg-types
        batch_id = 0
        for _ in range(len(list(ds_bpd))):
          eval_batch = get_batch(config, next(bpd_iter), scaler)
          bpd = likelihood_fn(score_model, eval_batch)[0]
          bpd = bpd.detach().cpu().numpy().reshape(-1)
          bpds.extend(bpd)
          logging.info(
            "ckpt: %d, repeat: %d, batch: %d, mean bpd: %6f" % (ckpt, repeat, batch_id, np.mean(np.asarray(bpds))))
          bpd_round_id = batch_id + len(ds_bpd) * repeat
          batch_id += 1
          # Save bits/dim to disk or Google Cloud Storage
          with tf.io.gfile.GFile(os.path.join(eval_dir,
                                              f"{config.eval.bpd_dataset}_ckpt_{ckpt}_bpd_{bpd_round_id}.npz"),
                                 "wb") as fout:
            io_buffer = io.BytesIO()
            np.savez_compressed(io_buffer, bpd)
            fout.write(io_buffer.getvalue())

    # Generate samples and compute IS/FID/KID when enabled
    if config.eval.enable_sampling:
      num_sampling_rounds = (config.eval.num_samples-1) // config.eval.batch_size + 1
      for r in range(num_sampling_rounds):
        #sampling_idx = r
        sampling_idx = np.random.randint(0, 10000000)
        samples = sampling_lib.get_samples(config, score_model, sampling_fn, ckpt, sampling_idx, this_sample_dir)
        save.save_image_(samples, sampling_idx, this_sample_dir)
        import sys
        sys.exit()

      samples_dir = tf.io.gfile.glob(os.path.join(this_sample_dir, "samples_*.npz"))
      # Use inceptionV3 for images with resolution higher than 256.
      inceptionv3 = config.data.image_size >= 256
      inception_model = evaluation.get_inception_model(inceptionv3=inceptionv3)
      for sample_name in samples_dir:
        sampling_idx = int(sample_name.split('/')[-1].split('_')[1].split('.')[0])
        samples = sampling_lib.get_samples(config, score_model, sampling_fn, ckpt, sampling_idx, this_sample_dir)
        latents = sampling_lib.get_latent(samples, inception_model, inceptionv3, this_sample_dir, sampling_idx)
        save.save_statistics(latents, sampling_idx, this_sample_dir)
      #train_ds, eval_ds, _ = datasets.get_dataset(config, batch_size=200,
      #                                                    uniform_dequantization=config.data.uniform_dequantization)
      #train_iter = iter(train_ds)
      train_iter = []
      sampling_lib.compute_fid_and_is(config, assetdir, inceptionv3, ckpt, train_iter, sample_dir=this_sample_dir)