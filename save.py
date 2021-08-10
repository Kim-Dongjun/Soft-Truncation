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

# Modified at 2021 by the authors of "Score Matching Model for Unbounded Data Score"



import tensorflow as tf
import os
import io
import numpy as np
import torch
from torchvision.utils import make_grid, save_image

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

def save_image_(samples, name, saving_dir):
    name = create_name('samples', name, 'png')
    if not os.path.exists(os.path.join(saving_dir, name)):
        samples = torch.tensor(samples).permute(0, 3, 1, 2) / 255.
        nrow = int(np.sqrt(samples.shape[0]))
        image_grid = make_grid(samples, nrow, padding=2)
        with tf.io.gfile.GFile(
                os.path.join(saving_dir, name), "wb") as fout:
            save_image(image_grid, fout)

def save_statistics(latents, name, saving_dir):
    name = create_name('statistics', name, 'npz')
    if not os.path.exists(os.path.join(saving_dir, name)):
        # Save latent represents of the Inception network to disk or Google Cloud Storage
        with tf.io.gfile.GFile(
                os.path.join(saving_dir, name), "wb") as fout:
            io_buffer = io.BytesIO()
            np.savez_compressed(
                io_buffer, pool_3=latents["pool_3"], logits=latents["logits"])
            fout.write(io_buffer.getvalue())

def save_bpd(config, bpd, ckpt, batch_id, ds_bpd, repeat, saving_dir):
    bpd_round_id = batch_id + len(ds_bpd) * repeat
    # Save bits/dim to disk or Google Cloud Storage
    with tf.io.gfile.GFile(os.path.join(saving_dir,
                                        f"{config.eval.bpd_dataset}_ckpt_{ckpt}_bpd_{bpd_round_id}.npz"),
                           "wb") as fout:
        io_buffer = io.BytesIO()
        np.savez_compressed(io_buffer, bpd)
        fout.write(io_buffer.getvalue())

def save_loss(all_losses, ckpt, saving_dir):
    # Save loss values to disk or Google Cloud Storage
    all_losses = np.asarray(all_losses)
    with tf.io.gfile.GFile(os.path.join(saving_dir, f"ckpt_{ckpt}_loss.npz"), "wb") as fout:
        io_buffer = io.BytesIO()
        np.savez_compressed(io_buffer, all_losses=all_losses, mean_loss=all_losses.mean())
        fout.write(io_buffer.getvalue())