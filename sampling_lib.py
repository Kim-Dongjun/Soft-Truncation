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

"""Utility functions for computing FID/Inception scores."""

import logging
import os
import io
import gc
import torch
import numpy as np
import tensorflow as tf
import tensorflow_gan as tfgan

import evaluation
from save import create_name
from cleanfid import fid as fid_calculator

def get_losses(config, eval_ds, scaler, eval_step, state):
    all_losses = []
    eval_iter = iter(eval_ds)  # pytype: disable=wrong-arg-types
    for i, batch in enumerate(eval_iter):
        eval_batch = torch.from_numpy(batch['image']._numpy()).to(config.device).float()
        eval_batch = eval_batch.permute(0, 3, 1, 2)
        eval_batch = scaler(eval_batch)
        eval_loss = eval_step(state, eval_batch)
        all_losses.append(eval_loss.item())
        if (i + 1) % 1000 == 0:
            logging.info("Finished %dth step loss evaluation" % (i + 1))
    return all_losses

def get_samples_manual(config, score_model, sampling_fn, name, sample_dir):
    if len(name.split('.')) == 1:
        name = name + '.npz'
    if not os.path.exists(os.path.join(sample_dir, name)):
        print("sampling start ...")
        # Directory to save samples. Different for each host to avoid writing conflicts
        tf.io.gfile.makedirs(sample_dir)
        samples, n = sampling_fn(score_model)
        samples = np.clip(samples.permute(0, 2, 3, 1).cpu().numpy() * 255., 0, 255).astype(np.uint8)
        samples = samples.reshape(
            (-1, config.data.image_size, config.data.image_size, config.data.num_channels))
        # Write samples to disk or Google Cloud Storage
        with tf.io.gfile.GFile(
                os.path.join(sample_dir, name), "wb") as fout:
            io_buffer = io.BytesIO()
            np.savez_compressed(io_buffer, samples=samples)
            fout.write(io_buffer.getvalue())
    else:
        try:
            samples = np.load(os.path.join(sample_dir, name))['samples']
            samples = samples[np.random.choice(samples.shape[0], 1000, replace=False)]
        except:
            try:
                samples = np.load(os.path.join(sample_dir, name))['sample']
                samples = samples[np.random.choice(samples.shape[0], 1000, replace=False)]
            except:
                samples = np.load(os.path.join(sample_dir, name))#['samples']
                samples = samples[np.random.choice(samples.shape[0], 1000, replace=False)]
    return samples

def get_samples(config, score_model, sampling_fn, ckpt, r, sample_dir):
    logging.info("sampling -- ckpt: %d, round: %d" % (ckpt, r))
    if not os.path.exists(os.path.join(sample_dir, f'samples_{r}.npz')):
        # Directory to save samples. Different for each host to avoid writing conflicts
        tf.io.gfile.makedirs(sample_dir)
        samples, n = sampling_fn(score_model, r)
        samples = np.clip(samples.permute(0, 2, 3, 1).cpu().numpy() * 255., 0, 255).astype(np.uint8)
        samples = samples.reshape(
            (-1, config.data.image_size, config.data.image_size, config.data.num_channels))
        # Write samples to disk or Google Cloud Storage
        with tf.io.gfile.GFile(
                os.path.join(sample_dir, f"samples_{r}.npz"), "wb") as fout:
            io_buffer = io.BytesIO()
            np.savez_compressed(io_buffer, samples=samples)
            fout.write(io_buffer.getvalue())
    else:
        samples = np.load(os.path.join(sample_dir, f"samples_{r}.npz"))['samples']
    return samples

def get_latent(samples, inception_model, inceptionv3, saving_dir, name, small_batch=128):
    latents = {}
    num = (samples.shape[0] - 1) // small_batch + 1
    name = create_name('statistics', name, 'npz')
    if not os.path.exists(os.path.join(saving_dir, name)):
        for k in range(num):
            # Force garbage collection before calling TensorFlow code for Inception network
            gc.collect()
            latents_temp = evaluation.run_inception_distributed(samples[small_batch * k:small_batch * (k + 1)],
                                                                inception_model,
                                                                inceptionv3=inceptionv3)
            if k == 0:
                latents['pool_3'] = latents_temp['pool_3']
                latents['logits'] = latents_temp['logits']
            else:
                latents['pool_3'] = tf.concat([latents['pool_3'], latents_temp['pool_3']], 0)
                latents['logits'] = tf.concat([latents['logits'], latents_temp['logits']], 0)
            # Force garbage collection again before returning to JAX code
            gc.collect()
    else:
        latents = ''
    return latents

def compute_fid_and_is(config, assetdir, inceptionv3, ckpt, dataset, name='/0', sample_dir='', latents=''):
    if config.data.dataset == 'CIFAR10':
        compute_fid_and_is_cifar10(config, assetdir, inceptionv3, ckpt, name=name, sample_dir=sample_dir, latents=latents)
    elif config.data.dataset in ['FFHQ', 'LSUN', 'CelebAHQ']:
        compute_fid_256(config, assetdir, inceptionv3, ckpt, dataset, name=name, sample_dir=sample_dir, latents=latents)
    else:
        raise NotImplementedError

def compute_fid_256(config, assetdir, inceptionv3, ckpt, dataset, name='/0', sample_dir='', latents=''):
    fids = fid_calculator.compute_fid(config=config, mode='clean', fdir1=sample_dir, dataset_name=config.data.dataset, assetdir=assetdir, dataset=dataset)

    logging.info(f"{sample_dir}_ckpt-%d_{name} --- FID: {fids}" % (ckpt))

    if len(name.split('.')) == 1:
        name = f'report_{name}.npz'
    else:
        name = f'report_{name.split(".")[0]}.npz'
    if not os.path.join(sample_dir, name):
        with tf.io.gfile.GFile(os.path.join(sample_dir, name),
                               "wb") as f:
            io_buffer = io.BytesIO()
            np.savez_compressed(io_buffer, fids=fids)
            f.write(io_buffer.getvalue())

def compute_fid_and_is_cifar10(config, assetdir, inceptionv3, ckpt, name='/0', sample_dir='', latents=''):
    # Compute inception scores, FIDs and KIDs.
    # Load all statistics that have been previously computed and saved for each host
    all_logits = []
    all_pools = []
    if latents == '':
        stats = tf.io.gfile.glob(os.path.join(sample_dir, "statistics_*.npz"))
        for stat_file in stats:
            with tf.io.gfile.GFile(stat_file, "rb") as fin:
                stat = np.load(fin)
                print("stat : ", stat)
                if not inceptionv3:
                    all_logits.append(stat["logits"])
                all_pools.append(stat["pool_3"])
    else:
        if not inceptionv3:
            all_logits.append(latents["logits"])
        all_pools.append(latents["pool_3"])

    if not inceptionv3:
        all_logits = np.concatenate(all_logits, axis=0)[:config.eval.num_samples]
    all_pools = np.concatenate(all_pools, axis=0)[:config.eval.num_samples]

    if config.eval.num_samples < 50000:
        random_indices = np.random.choice(len(all_pools), config.eval.num_samples, replace=False)
    else:
        random_indices = np.arange(len(all_pools))
    if not inceptionv3:
        all_logits = all_logits[random_indices]
    all_pools = all_pools[random_indices]

    print("all logits length : ", len(all_logits))
    assert len(all_logits) == config.eval.num_samples

    # Load pre-computed dataset statistics.
    data_stats = evaluation.load_dataset_stats(config, assetdir)
    data_pools = data_stats["pool_3"]

    # Compute FID/KID/IS on all samples together.
    if not inceptionv3:
        inception_score = tfgan.eval.classifier_score_from_logits(all_logits)
    else:
        inception_score = -1

    fid = tfgan.eval.frechet_classifier_distance_from_activations(
        data_pools, all_pools)
    # Hack to get tfgan KID work for eager execution.
    tf_data_pools = tf.convert_to_tensor(data_pools)
    tf_all_pools = tf.convert_to_tensor(all_pools)
    kid = tfgan.eval.kernel_classifier_distance_from_activations(
        tf_data_pools, tf_all_pools).numpy()
    del tf_data_pools, tf_all_pools
    name = name.split('/')[-1]
    logging.info(
        f"{sample_dir}_ckpt-%d_{name} --- inception_score: %.6e, FID: %.6e, KID: %.6e" % (
            ckpt, inception_score, fid, kid))

    if len(name.split('.')) == 1:
        name = f'report_{name}.npz'
    else:

        name = f'report_{name.split(".")[0]}.npz'
    if not os.path.join(sample_dir, name):
        with tf.io.gfile.GFile(os.path.join(sample_dir, name),
                               "wb") as f:
            io_buffer = io.BytesIO()
            np.savez_compressed(io_buffer, IS=inception_score, fid=fid, kid=kid)
            f.write(io_buffer.getvalue())