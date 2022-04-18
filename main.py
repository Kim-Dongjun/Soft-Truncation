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

"""Training and evaluation"""

import torch
import run_lib
from absl import app
from absl import flags
from ml_collections.config_flags import config_flags
import logging
import os
import tensorflow as tf

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file(
  "config", None, "Training configuration.", lock_config=True)
flags.DEFINE_string("workdir", None, "Work directory.")
flags.DEFINE_enum("mode", None, ["train", "eval"], "Running mode: train or eval")
flags.DEFINE_string("assetdir", os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))) +
                    "/assets/stats/", "The folder name for storing evaluation results")
flags.DEFINE_string("eval_folder", "eval",
                    "The folder name for storing evaluation results")
flags.mark_flags_as_required(["workdir", "config", "mode"])


def main(argv):
  tf.io.gfile.makedirs(FLAGS.workdir)
  with open(os.path.join(FLAGS.workdir, 'config.txt'), 'w') as f:
    # f.write(str(FLAGS.config.to_dict()))
    for k, v in FLAGS.config.to_dict().items():
      f.write(str(k) + '\n')
      print(type(v))
      if type(v) == dict:
        for k2, v2 in v.items():
          f.write('> ' + str(k2) + ': ' + str(v2) + '\n')
      f.write('\n\n')
  if FLAGS.mode == "train":
    # Create the working directory
    tf.io.gfile.makedirs(FLAGS.workdir)
    # Set logger so that it outputs to both console and file
    # Make logging work for both disk and Google Cloud Storage
    if os.path.exists(os.path.join(FLAGS.workdir, 'stdout.txt')):
      gfile_stream = open(os.path.join(FLAGS.workdir, 'stdout.txt'), 'a')
    else:
      gfile_stream = open(os.path.join(FLAGS.workdir, 'stdout.txt'), 'w')
    handler = logging.StreamHandler(gfile_stream)
    formatter = logging.Formatter('%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
    handler.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler)
    logger.setLevel('INFO')
    # Run the training pipeline
    run_lib.train(FLAGS.config, FLAGS.workdir, FLAGS.assetdir)
  elif FLAGS.mode == "eval":
    eval_dir = os.path.join(FLAGS.workdir, FLAGS.eval_folder)
    tf.io.gfile.makedirs(eval_dir)
    stdout_name = 'evaluation_history'
    if os.path.exists(os.path.join(FLAGS.workdir, f'{stdout_name}.txt')):
      gfile_stream = open(os.path.join(FLAGS.workdir, f'{stdout_name}.txt'), 'a')
    else:
      gfile_stream = open(os.path.join(FLAGS.workdir, f'{stdout_name}.txt'), 'w')
    handler = logging.StreamHandler(gfile_stream)
    formatter = logging.Formatter('%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
    handler.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler)
    logger.setLevel('INFO')
    # Run the evaluation pipeline
    run_lib.evaluate(FLAGS.config, FLAGS.workdir, FLAGS.assetdir, FLAGS.eval_folder)
  else:
    raise ValueError(f"Mode {FLAGS.mode} not recognized.")


if __name__ == "__main__":
  app.run(main)
