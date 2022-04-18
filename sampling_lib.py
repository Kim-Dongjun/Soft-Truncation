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

"""Utility functions for computing FID/Inception scores."""

import logging
import os
import io
import torch
import numpy as np
import gc
import evaluation
import utils
import tensorflow as tf
from torchvision.utils import make_grid, save_image

def get_dir_name(config, sample_dir, step):
    if config.sampling.method == 'pc':
        dir_name = os.path.join(sample_dir, f"iter_{step}_{config.sampling.truncation_time}_{config.sampling.noise_removal}_{config.sampling.predictor}_{config.sampling.corrector}_{config.sampling.snr}")
    else:
        dir_name = os.path.join(sample_dir, f"iter_{step}_{config.sampling.truncation_time}_{config.sampling.noise_removal}")
    return dir_name

def get_samples(config, score_model, state, sampling_fn, step, r, sample_dir):
    logging.info("sampling -- ckpt step: %d, round: %d" % (step, r))
    dir_name = get_dir_name(config, sample_dir, step)
    tf.io.gfile.makedirs(dir_name)
    if not os.path.exists(os.path.join(dir_name, f'samples_{r}.npz')):
        samples, n = sampling_fn(score_model)
        samples = np.clip(samples.permute(0, 2, 3, 1).cpu().numpy() * 255., 0, 255).astype(np.uint8)
        samples = samples.reshape((-1, config.data.image_size, config.data.image_size, config.data.num_channels))
        # Write samples to disk or Google Cloud Storage
        with tf.io.gfile.GFile(
                os.path.join(dir_name, f"samples_{r}.npz"), "wb") as fout:
            io_buffer = io.BytesIO()
            np.savez_compressed(io_buffer, samples=samples)
            fout.write(io_buffer.getvalue())
        nrow = int(np.sqrt(samples.shape[0]))
        image_grid = make_grid(torch.tensor(samples).permute(0, 3, 1, 2) / 255., nrow, padding=2)
        with tf.io.gfile.GFile(
                os.path.join(dir_name, f"sample_{r}.png"), "wb") as fout:
            save_image(image_grid, fout)
    else:
        samples = np.load(os.path.join(dir_name, f"samples_{r}.npz"))['samples']
    return samples

def get_latents(config, samples, inception_model, inceptionv3, step, r, sample_dir, small_batch=128):
    latents = {}
    num = (samples.shape[0] - 1) // small_batch + 1
    name = utils.create_name('statistics', r, 'npz')
    dir_name = get_dir_name(config, sample_dir, step)
    # samples = torch.tensor(samples, device=inception_model.device)
    if not os.path.exists(os.path.join(dir_name, name)):
        for k in range(num):
            # Force garbage collection before calling TensorFlow code for Inception network
            gc.collect()
            latents_temp = evaluation.run_inception_distributed(samples[small_batch * k:small_batch * (k + 1)],
                                                                inception_model,
                                                                inceptionv3=inceptionv3)
            if k == 0:
                latents['pool_3'] = latents_temp['pool_3']
                if not inceptionv3:
                    latents['logits'] = latents_temp['logits']
            else:
                latents['pool_3'] = tf.concat([latents['pool_3'], latents_temp['pool_3']], 0)
                if not inceptionv3:
                    latents['logits'] = tf.concat([latents['logits'], latents_temp['logits']], 0)
            # Force garbage collection again before returning to JAX code
            gc.collect()
    else:
        latents = ''
    return latents

def save_statistics(config, latents, inceptionv3, step, r, sample_dir):
    name = utils.create_name('statistics', r, 'npz')
    dir_name = get_dir_name(config, sample_dir, step)
    if not os.path.exists(os.path.join(dir_name, name)):
        # Save latent represents of the Inception network to disk or Google Cloud Storage
        with tf.io.gfile.GFile(
                os.path.join(dir_name, name), "wb") as fout:
            io_buffer = io.BytesIO()
            if not inceptionv3:
                np.savez_compressed(
                    io_buffer, pool_3=latents["pool_3"], logits=latents["logits"])
            else:
                np.savez_compressed(
                    io_buffer, pool_3=latents["pool_3"])
            fout.write(io_buffer.getvalue())