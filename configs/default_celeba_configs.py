import ml_collections
import torch


def get_default_configs():
  config = ml_collections.ConfigDict()
  # training
  config.training = training = ml_collections.ConfigDict()
  config.training.batch_size = 128
  training.n_iters = 1300001
  training.snapshot_freq = 50000
  training.log_freq = 50
  training.eval_freq = 100
  ## store additional checkpoints for preemption in cloud computing environments
  training.snapshot_freq_for_preemption = 10000
  ## produce samples at each snapshot.
  training.snapshot_sampling = True
  training.likelihood_weighting = False
  training.continuous = True
  training.reduce_mean = False
  training.importance_sampling = True
  training.unbounded_parametrization = False
  training.ddpm_score = True
  training.st = False
  training.truncation_time = 1e-5
  training.num_train_data = 162770
  training.reconstruction_loss = False
  training.stabilizing_constant = 1e-3
  training.whatever_sampling = False
  training.mixed = False
  training.ddpm_weight = 0.01
  training.balanced = False

  # sampling
  config.sampling = sampling = ml_collections.ConfigDict()
  sampling.n_steps_each = 1
  sampling.noise_removal = True
  sampling.probability_flow = False
  sampling.snr = 0.17
  sampling.batch_size = 512
  sampling.truncation_time = 1e-5
  sampling.sample_more = True

  # evaluation
  config.eval = evaluate = ml_collections.ConfigDict()
  evaluate.begin_ckpt = 1
  evaluate.end_ckpt = 26
  evaluate.batch_size = 1024
  evaluate.enable_sampling = False
  evaluate.num_samples = 50000
  evaluate.enable_loss = True
  evaluate.enable_bpd = False
  evaluate.bpd_dataset = 'test'
  evaluate.num_test_data = 19962
  evaluate.residual = True
  evaluate.lambda_ = 0.0
  evaluate.probability_flow = True
  evaluate.nelbo_iter = 0
  evaluate.nll_iter = 0

  # data
  config.data = data = ml_collections.ConfigDict()
  data.dataset = 'CELEBA'
  data.image_size = 64
  data.random_flip = True
  data.dequantization = 'none'
  data.centered = False
  data.num_channels = 3

  # model
  config.model = model = ml_collections.ConfigDict()
  model.sigma_max = 90.
  model.sigma_min = 0.01
  model.num_scales = 1000
  model.beta_min = 0.1
  model.beta_max = 20.
  model.dropout = 0.1
  model.embedding_type = 'fourier'
  model.auxiliary_resblock = True
  model.attention = True
  model.fourier_feature = False
  model.lsgm = False

  # optimization
  config.optim = optim = ml_collections.ConfigDict()
  optim.weight_decay = 0
  optim.optimizer = 'Adam'
  optim.lr = 2e-4
  optim.beta1 = 0.9
  optim.eps = 1e-8
  optim.warmup = 5000
  optim.grad_clip = 1.
  optim.num_micro_batch = 1
  optim.amsgrad = False

  config.seed = 42
  config.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

  return config