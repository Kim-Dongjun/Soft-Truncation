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
"""Return training and evaluation/test datasets from config files."""
import torch
import tensorflow as tf
import tensorflow_datasets as tfds
import logging
import socket, os, natsort
import torchvision.transforms as transforms
from PIL import Image

def _data_transforms_generic(size):
  train_transform = transforms.Compose([
    transforms.Resize(size),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
  ])

  valid_transform = transforms.Compose([
    transforms.Resize(size),
    transforms.ToTensor(),
  ])

  return train_transform, valid_transform

class ImagenetDataSet(torch.utils.data.Dataset):
  def __init__(self, main_dir, transform):
    self.main_dir = main_dir
    self.transform = transform
    all_imgs = os.listdir(main_dir)
    self.total_imgs = natsort.natsorted(all_imgs)

  def __len__(self):
    return len(self.total_imgs)

  def __getitem__(self, idx):
    img_loc = os.path.join(self.main_dir, self.total_imgs[idx])
    image = Image.open(img_loc).convert("RGB")
    tensor_image = self.transform(image)
    return tensor_image

def get_data_scaler(config):
  """Data normalizer. Assume data are always in [0, 1]."""
  if config.data.centered:
    # Rescale to [-1, 1]
    return lambda x: x * 2. - 1.
  else:
    return lambda x: x


def get_data_inverse_scaler(config):
  """Inverse data normalizer."""
  if config.data.centered:
    # Rescale [-1, 1] to [0, 1]
    return lambda x: (x + 1.) / 2.
  else:
    return lambda x: x


def crop_resize(image, resolution):
  """Crop and resize an image to the given resolution."""
  crop = tf.minimum(tf.shape(image)[0], tf.shape(image)[1])
  h, w = tf.shape(image)[0], tf.shape(image)[1]
  image = image[(h - crop) // 2:(h + crop) // 2,
          (w - crop) // 2:(w + crop) // 2]
  image = tf.image.resize(
    image,
    size=(resolution, resolution),
    antialias=True,
    method=tf.image.ResizeMethod.BICUBIC)
  return tf.cast(image, tf.uint8)


def resize_small(image, resolution):
  """Shrink an image to the given resolution."""
  h, w = image.shape[0], image.shape[1]
  ratio = resolution / min(h, w)
  #h = tf.round(h * ratio, tf.int32)
  #w = tf.round(w * ratio, tf.int32)
  h = int(h * ratio)
  w = int(w * ratio)
  return tf.image.resize(image, [h, w], antialias=True)


def central_crop(image, size):
  """Crop the center of an image to the given size."""
  top = (image.shape[0] - size) // 2
  left = (image.shape[1] - size) // 2
  return tf.image.crop_to_bounding_box(image, top, left, size, size)


def get_batch(config, data_iter, data):
  try:
    batch = get_batch_(config, next(data_iter))
  except:
    logging.info('New Epoch Start')
    data_iter = iter(data)
    batch = get_batch_(config, next(data_iter))
  return batch, data_iter

def get_batch_(config, batch):
  if isinstance(batch, torch.ByteTensor):
    batch = batch#.to(config.device).float().permute(0, 3, 1, 2) / 255.
  else:
    if config.data.dataset in ['STL10', 'CIFAR100']:
      batch = batch[0]#.to(config.device)
    elif config.data.dataset in ['IMAGENET32', 'IMAGENET64']:
      batch = batch#.to(config.device)
    else:
      batch = torch.from_numpy(batch['image']._numpy()).float()#.to(config.device).float()
      batch = batch.permute(0, 3, 1, 2)
  assert batch.shape == (batch.shape[0], config.data.num_channels, config.data.image_size, config.data.image_size)

  return batch.to(config.device)

def check_dataset(config, train_ds, eval_ds):
  if config.data.dataset in ['IMAGENET32', 'IMAGENET64']:
    num_train_data = len(train_ds.dataset)
    num_eval_data = len(eval_ds.dataset)
    assert num_train_data == config.training.num_train_data and num_eval_data == config.eval.num_test_data

def get_dataset(config):
  if config.data.dataset in ['IMAGENET32', 'STL10']:
    train_ds, eval_ds = get_dataset_from_torch(config)
  else:
    train_ds = get_dataset_from_tf(config, evaluation=False)
    eval_ds = get_dataset_from_tf(config, evaluation=True)
  check_dataset(config, train_ds, eval_ds)
  return train_ds, eval_ds

def get_dataset_from_torch(config):
  if config.data.dataset == 'IMAGENET32':
    ip = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    ip.connect(("8.8.8.8", 80))
    ip = ip.getsockname()[0]
    if str(ip) in ['143.248.82.29', '143.248.84.89']:
      train_data_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))) + \
                        '/downloaded_data/IMAGENET32/small/train_32x32/'
      eval_data_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))) + \
                       '/downloaded_data/IMAGENET32/small/valid_32x32/'
    else:
      train_data_path = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))) + \
                        '/data/IMAGENET32/small/train_32x32/'
      eval_data_path = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))) + \
                       '/data/IMAGENET32/small/valid_32x32/'
    train_transform, val_transform = _data_transforms_generic(config.data.image_size)
    train_data = ImagenetDataSet(train_data_path, train_transform)
    eval_data = ImagenetDataSet(eval_data_path, val_transform)

    train_ds = torch.utils.data.DataLoader(train_data, batch_size=config.training.batch_size, shuffle=True,
                                           pin_memory=True, num_workers=8, drop_last=True)
    eval_ds = torch.utils.data.DataLoader(eval_data, batch_size=config.eval.batch_size, shuffle=False, pin_memory=True,
                                          num_workers=1, drop_last=False)

  elif config.data.dataset == 'STL10':
    ip = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    ip.connect(("8.8.8.8", 80))
    ip = ip.getsockname()[0]
    print("ip: ", ip)
    import torchvision.datasets as datasets
    import torchvision.transforms as transforms
    Dt = datasets.STL10
    transform = transforms.Compose([
      transforms.Resize(48),
      transforms.RandomHorizontalFlip(),
      transforms.ToTensor()
    ])

    if str(ip) in ['143.248.82.29', '143.248.84.89']:
      data_dir = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))) + f'/tensorflow_datasets/'
      logging.info(
        f'data path: {os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))) + "/tensorflow_datasets/"}')
    else:
      data_dir = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))) + f'/data/'
      logging.info(
        f'data path: {os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))) + "/data/"}')

    train_dataset = Dt(root=data_dir + '/STL10/', split='train+unlabeled', transform=transform, download=True)
    val_dataset = Dt(root=data_dir + '/STL10/', split='test', transform=transform)

    train_ds = torch.utils.data.DataLoader(
      train_dataset,
      batch_size=config.training.batch_size, num_workers=8, pin_memory=True, shuffle=True, drop_last=True)

    eval_ds = torch.utils.data.DataLoader(
      val_dataset,
      batch_size=config.eval.batch_size, shuffle=False, pin_memory=True, num_workers=1, drop_last=False)

  return train_ds, eval_ds

def get_dataset_from_tf(config, evaluation=False):
  """Create data loaders for training and evaluation.

  Args:
    config: A ml_collection.ConfigDict parsed from config files.
    evaluation: If `True`, fix number of epochs to 1.

  Returns:
    train_ds, eval_ds, dataset_builder.
  """
  # Compute batch size for this worker.
  batch_size = config.training.batch_size if not evaluation else config.eval.batch_size
  if batch_size % torch.cuda.device_count() != 0:
    raise ValueError(f'Batch sizes ({batch_size} must be divided by the number of devices')

  # Reduce this when image resolution is too large and data pointer is stored
  shuffle_buffer_size = 10000
  prefetch_size = tf.data.experimental.AUTOTUNE
  num_epochs = None if not evaluation else 1

  # Create dataset builders for each dataset.
  if config.data.dataset == 'CIFAR10':
    ip = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    ip.connect(("8.8.8.8", 80))
    ip = ip.getsockname()[0]
    if str(ip) in ['143.248.82.29', '143.248.84.89', '143.248.80.183', '143.248.80.109']:
      dataset_builder = tfds.builder('cifar10', data_dir=os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))) + f'/tensorflow_datasets/')
    else:
      dataset_builder = tfds.builder('cifar10', data_dir=os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))) + f'/data/')
    #dataset_builder = tfds.builder('cifar10')
    train_split_name = 'train'
    eval_split_name = 'test'

    def resize_op(img):
      img = tf.image.convert_image_dtype(img, tf.float32)
      return tf.image.resize(img, [config.data.image_size, config.data.image_size], antialias=True)

  elif config.data.dataset == 'SVHN':
    dataset_builder = tfds.builder('svhn_cropped')
    train_split_name = 'train'
    eval_split_name = 'test'

    def resize_op(img):
      img = tf.image.convert_image_dtype(img, tf.float32)
      return tf.image.resize(img, [config.data.image_size, config.data.image_size], antialias=True)

  elif config.data.dataset == 'CELEBA':
    ip = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    ip.connect(("8.8.8.8", 80))
    ip = ip.getsockname()[0]
    print("ip: ", ip)
    if str(ip) in ['143.248.82.29', '143.248.84.89']:
      dataset_builder = tfds.builder('celeb_a', data_dir=os.path.dirname(
          os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))) + f'/tensorflow_datasets/')
      logging.info(f'data path: {os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))) + "/tensorflow_datasets/"}')
    else:
      dataset_builder = tfds.builder('celeb_a', data_dir=os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))) + f'/data/')
      logging.info(f'data path: {os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))) + "/data/"}')
    train_split_name = 'train'
    eval_split_name = 'test'

    def resize_op(img):
      img = tf.image.convert_image_dtype(img, tf.float32)
      img = central_crop(img, 140)
      img = resize_small(img, config.data.image_size)
      return img

  elif config.data.dataset == 'LSUN':
    dataset_builder = tfds.builder(f'lsun/{config.data.category}')
    train_split_name = 'train'
    eval_split_name = 'validation'

    if config.data.image_size == 128:
      def resize_op(img):
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = resize_small(img, config.data.image_size)
        img = central_crop(img, config.data.image_size)
        return img

    else:
      def resize_op(img):
        img = crop_resize(img, config.data.image_size)
        img = tf.image.convert_image_dtype(img, tf.float32)
        return img

  elif config.data.dataset in ['FFHQ', 'CelebAHQ']:
    dataset_builder = tf.data.TFRecordDataset(config.data.tfrecords_path)
    train_split_name = eval_split_name = 'train'

  else:
    raise NotImplementedError(
      f'Dataset {config.data.dataset} not yet supported.')

  # Customize preprocess functions for each dataset.
  if config.data.dataset in ['FFHQ', 'CelebAHQ']:
    def preprocess_fn(d):
      sample = tf.io.parse_single_example(d, features={
        'shape': tf.io.FixedLenFeature([3], tf.int64),
        'data': tf.io.FixedLenFeature([], tf.string)})
      data = tf.io.decode_raw(sample['data'], tf.uint8)
      data = tf.reshape(data, sample['shape'])
      data = tf.transpose(data, (1, 2, 0))
      img = tf.image.convert_image_dtype(data, tf.float32)
      if config.data.random_flip and not evaluation:
        img = tf.image.random_flip_left_right(img)
      return dict(image=img, label=None)

  else:
    def preprocess_fn(d):
      """Basic preprocessing function scales data to [0, 1) and randomly flips."""
      img = resize_op(d['image'])
      if config.data.random_flip and not evaluation:
        img = tf.image.random_flip_left_right(img)

      return dict(image=img, label=d.get('label', None))

  def create_dataset(dataset_builder, split, batch_size):
    dataset_options = tf.data.Options()
    dataset_options.experimental_optimization.map_parallelization = True
    dataset_options.experimental_threading.private_threadpool_size = 48
    dataset_options.experimental_threading.max_intra_op_parallelism = 1
    read_config = tfds.ReadConfig(options=dataset_options)
    if isinstance(dataset_builder, tfds.core.DatasetBuilder):
      dataset_builder.download_and_prepare()
      ds = dataset_builder.as_dataset(
        split=split, shuffle_files=True, read_config=read_config)
    else:
      ds = dataset_builder.with_options(dataset_options)
    ds = ds.repeat(count=num_epochs)
    ds = ds.shuffle(shuffle_buffer_size)
    ds = ds.map(preprocess_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=(not evaluation))
    return ds.prefetch(prefetch_size)

  if evaluation:
    data = create_dataset(dataset_builder, eval_split_name, config.eval.batch_size)
  else:
    data = create_dataset(dataset_builder, train_split_name, config.training.batch_size)
  return data
