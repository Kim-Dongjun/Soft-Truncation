import os
import random
from tqdm import tqdm
from glob import glob
import torch
import numpy as np
from scipy import linalg
import tensorflow as tf
import os
import io
import torchvision
from torchvision.utils import make_grid, save_image
import logging
from datasets import get_batch
import datasets
import time

import cleanfid
from cleanfid.utils import *
from cleanfid.features import *
from cleanfid.resize import *


"""
Compute the FID score given the mu, sigma of two sets
"""
def frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.
    Returns:
    --   : The Frechet Distance.
    """
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)
    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    logging.info(f'stats : {diff.dot(diff)}, {np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean}')

    return (diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)


"""
Compute the KID score given the sets of features
"""
def kernel_distance(feats1, feats2, num_subsets=100, max_subset_size=1000):
    n = feats1.shape[1]
    m = min(min(feats1.shape[0], feats2.shape[0]), max_subset_size)
    t = 0
    for _subset_idx in range(num_subsets):
        x = feats2[np.random.choice(feats2.shape[0], m, replace=False)]
        y = feats1[np.random.choice(feats1.shape[0], m, replace=False)]
        a = (x @ x.T / n + 1) ** 3 + (y @ y.T / n + 1) ** 3
        b = (x @ y.T / n + 1) ** 3
        t += (a.sum() - np.diag(a).sum()) / (m - 1) - b.sum() * 2 / m
    kid = t / num_subsets / m
    return float(kid)


"""
Compute the inception features for a batch of images
"""
def get_batch_features(batch, model, device):
    print(f"input shape: {batch.shape}, input max: {torch.max(batch)}")
    with torch.no_grad():
        feat = model(batch.to(device).permute(0,3,1,2))
    return feat.detach().cpu().numpy()

"""
Compute the inception features for a list of files
"""
def get_files_features(fdir, l_files, model=None, num_workers=12,
                       batch_size=128, device=torch.device("cuda"),
                       mode="clean", custom_fn_resize=None, 
                       description=""):
    # define the model if it is not specified
    if model is None:
        model = build_feature_extractor(mode, device)
    
    # build resizing function based on options
    if custom_fn_resize is not None:
        fn_resize = custom_fn_resize
    else:
        fn_resize = build_resizer(mode)

    l_feats = []
    transforms = torchvision.transforms.ToTensor()
    file_itr = 0
    for file in l_files:
        save_filename = f'np_feats_{mode}_{file.split("/")[-1].split(".")[0].split("_")[1]}.npz'
        if not os.path.exists(os.path.join(fdir, f'features/{save_filename}')):
            img_np = np.load(file)['samples']#.reshape(-1,256,256,3)
            assert img_np.shape == (img_np.shape[0], img_np.shape[2], img_np.shape[2], 3)
            assert np.max(img_np) > 2.
            print("generated: ", img_np.shape)
            itr = 0
            batch = torch.zeros((img_np.shape[0], 299, 299, 3), device=device)
            for img in img_np:
                img_resized = fn_resize(img)
                if img_resized.dtype == "uint8":
                    img_t = transforms(np.array(img_resized)) * 255
                elif img_resized.dtype == "float32":
                    img_t = transforms(img_resized)
                img_t = img_t.permute(1,2,0)
                batch[itr] = img_t
                if itr == 0:
                    print("--------------------------")
                    print(f"file iter : {file_itr}/{len(l_files)}")
                    file_itr += 1
                itr += 1
            #batch = torch.clip(batch, 0., 255.)
            if torch.max(batch) < 2.:
                batch = batch * 255.
            assert torch.max(batch) > 2.
            print("batch: ", torch.min(batch), torch.max(batch), batch.shape)
            if batch.shape[0] > 1024:
                temp = []
                temp.extend(get_batch_features(batch[:1024], model, device))
                temp.extend(get_batch_features(batch[1024:], model, device))
                l_feats.append(temp)
            else:
                l_feats.append(get_batch_features(batch, model, device))
            with tf.io.gfile.GFile(
                    os.path.join(fdir, f'features/{save_filename}'),
                    "wb") as fout:
                io_buffer = io.BytesIO()
                np.savez_compressed(io_buffer, np_feats=l_feats[-1])
                fout.write(io_buffer.getvalue())
        else:
            feat = np.load(os.path.join(fdir, f'features/{save_filename}'))['np_feats']
            l_feats.append(feat)
        print(f"l_feats stats : {np.min(l_feats[-1]), np.max(l_feats[-1]), np.mean(l_feats[-1])}")
        print("l_feats size : ", len(l_feats))
    np_feats = np.concatenate(l_feats)

    return np_feats

"""
Compute the inception features for a folder of features
"""
def get_folder_features(fdir, model=None, num_workers=12, num=None,
                        shuffle=False, seed=0, batch_size=128, device=torch.device("cuda"),
                        mode="clean", custom_fn_resize=None, description=""):
    # get all relevant files in the dataset
    files = sorted([file for ext in EXTENSIONS
                    for file in glob(os.path.join(fdir, f"*.{ext}"))])
    files_tmp = []

    for file in files:
        print(file)
        print(file.split('/')[-1])
        if len(file.split('/')[-1].split('_')) == 2:
            files_tmp.append(file)

    files = files_tmp

    if num is not None:
        if shuffle:
            random.seed(seed)
            random.shuffle(files)
        files = files[:num]
    np_feats = get_files_features(fdir, files, model, num_workers=num_workers,
                                  batch_size=batch_size, device=device,
                                  mode=mode,
                                  custom_fn_resize=custom_fn_resize,
                                  description=description)
    return np_feats

"""
Compute the FID score given the inception features stack
"""
def fid_from_feats(feats1, feats2):
    mu1, sig1 = np.mean(feats1, axis=0), np.cov(feats1, rowvar=False)
    mu2, sig2 = np.mean(feats2, axis=0), np.cov(feats2, rowvar=False)
    return frechet_distance(mu1, sig1, mu2, sig2)

"""
Computes the FID score for a folder of images for a specific dataset 
and a specific resolution
"""
def fid_folder(fdir, dataset, dataset_name, dataset_res, dataset_split,
               model=None, mode="clean", num_workers=12,
               batch_size=128, device=torch.device("cuda"), assetdir='', config=None, dequantization=True,
               num_data=1000):
    # define the model if it is not specified
    if model is None:
        model = build_feature_extractor(mode, device)
    # Load reference FID statistics (download if needed)
    if dataset == []:
        ref_mu, ref_sigma, stats = get_statistics(config, assetdir, mode)
    else:
        ref_mu, ref_sigma = get_statistics_from_dataset(fdir, config, dataset, mode, device, dequantization)
    #ref_mu, ref_sigma = get_reference_statistics(dataset_name, dataset_res,
    #                                mode=mode, seed=0, split=dataset_split)
    fbname = os.path.basename(fdir)
    # get all inception features for folder images
    tf.io.gfile.makedirs(os.path.join(fdir, 'features/'))
    if os.path.exists(os.path.join(fdir, f'features/np_feats_{mode}.npz')):
        np_feats = np.load(os.path.join(fdir, f'features/np_feats_{mode}.npz'))['np_feats']
    else:
        np_feats = get_folder_features(fdir, model, num_workers=num_workers,
                                       batch_size=batch_size, device=device,
                                       mode=mode, description=f"FID {fbname} : ")
        with tf.io.gfile.GFile(
                os.path.join(fdir, f'features/np_feats_{mode}.npz'), "wb") as fout:
            io_buffer = io.BytesIO()
            np.savez_compressed(io_buffer, np_feats=np_feats)
            fout.write(io_buffer.getvalue())
    num_samples = 50000
    if num_data != None:
        num_samples = num_data
    np_feats = np_feats[:num_samples]
    num_samples = np_feats.shape[0]
    logging.info(f'Number of samples : {num_samples}')
    print("shape of np_feats : ", np_feats.shape)
    mu = np.mean(np_feats, axis=0)
    sigma = np.cov(np_feats, rowvar=False)
    print("mu stats : ", np.min(mu), np.max(mu))
    print("ref mu stats : ", np.min(ref_mu), np.max(ref_mu))

    print("sigma stats : ", np.min(sigma), np.max(sigma))
    print("ref sigma stats : ", np.min(ref_sigma), np.max(ref_sigma))
    fid = frechet_distance(mu, sigma, ref_mu, ref_sigma)
    return fid

def get_statistics(config, assetdir, mode):
    from evaluation import load_dataset_stats
    stats = load_dataset_stats(config, assetdir, mode)
    print("read statistics from asset")
    if 'mu' in list(stats.keys()):
        return stats["mu"], stats["sigma"], []
    else:
        try:
            stats = stats['real_feats']
        except:
            stats = stats['np_feats']
        return np.mean(stats, axis=0), np.cov(stats, rowvar=False), stats

def get_statistics_from_dataset(fdir, config, dataset, mode, device, dequantization):
    if not os.path.exists(os.path.join(fdir, f'features/{config.data.dataset}_stats.npz')):
        model = build_feature_extractor(mode, device)
        scaler = datasets.get_data_scaler(config)
        # build resizing function based on options
        fn_resize = build_resizer(mode)
        transforms = torchvision.transforms.ToTensor()

        real_feats = []
        if config.data.dataset.lower() == 'celeba':
            data_num = 162770
        elif config.data.dataset.lower() == 'imagenet32':
            data_num = 1281149
        elif config.data.dataset.lower() == 'cifar100':
            data_num = 50000
        elif config.data.dataset.lower() == 'stl10':
            data_num = 105000
        batch_size = 2000
        left = data_num
        itr = 0
        while left > 0:

            if not os.path.exists(os.path.join(fdir, f'features/real_feats_{mode}_{(itr * batch_size)}-{(itr+1)*batch_size}.npz')):
                if not config.data.dataset in ['STL10', 'CIFAR100']:
                    if config.data.dataset in ['IMAGENET32', 'IMAGENET64']:
                        imgs_np = next(dataset).permute(0,2,3,1).numpy() * 255.
                    else:
                        imgs_np = next(dataset)['image']._numpy()
                    if np.max(imgs_np) < 2:
                        imgs_np = imgs_np * 255.
                else:
                    imgs_np = (next(dataset)[0].permute(0,2,3,1).cpu().detach().numpy() * 255.).astype(np.uint8)
                assert imgs_np.shape == (imgs_np.shape[0], config.data.image_size, config.data.image_size, 3)
                if left < batch_size:
                    imgs_np = imgs_np[:left]
                print("loaded image range: ", np.min(imgs_np), np.max(imgs_np))
                assert np.max(imgs_np) > 2
                print("real: ", imgs_np.shape)
                batch = torch.zeros((imgs_np.shape[0], 299, 299, 3), device=config.device)
                i = 0
                for img in imgs_np:
                    img_resized = fn_resize(img)
                    if img_resized.dtype == "uint8":
                        if config.data.dataset != 'STL10':
                            img_t = transforms(np.array(img_resized)) * 255
                        else:
                            img_t = transforms(np.array(img_resized))
                    elif img_resized.dtype == "float32":
                        img_t = transforms(img_resized)
                    img_t = img_t.permute(1,2,0)
                    batch[i] = img_t
                    i += 1
                print("batch stats : ", torch.min(batch), torch.max(batch), batch.shape)
                print("--------------------------")
                print(f"left data : {left}")
                if torch.max(batch) < 2.:
                    batch = batch * 255.
                assert torch.max(batch) > 2.
                real_feats.append(get_batch_features(batch, model, device))
                print(f"real_feats stats : {np.min(real_feats[-1]), np.max(real_feats[-1]), np.mean(real_feats[-1])}")
                tf.io.gfile.makedirs(os.path.join(fdir, 'features/'))
                with tf.io.gfile.GFile(
                        os.path.join(fdir, f'features/real_feats_{mode}_{(itr * batch_size)}-{(itr+1)*batch_size}.npz'), "wb") as fout:
                    io_buffer = io.BytesIO()
                    np.savez_compressed(io_buffer, real_feats=np.array(real_feats[-1]))
                    fout.write(io_buffer.getvalue())
            else:
                print("i: ", itr)
                real_feats.append(np.load(os.path.join(fdir, f'features/real_feats_{mode}_{(itr * batch_size)}-{(itr+1)*batch_size}.npz'))['real_feats'])
            itr += 1
            left -= batch_size
        real_feats = np.concatenate(real_feats, axis=0)
        tf.io.gfile.makedirs(os.path.join(fdir, 'features/'))
        with tf.io.gfile.GFile(
                os.path.join(fdir, f'features/real_feats_{mode}.npz'), "wb") as fout:
            io_buffer = io.BytesIO()
            np.savez_compressed(io_buffer, real_feats=real_feats)
            fout.write(io_buffer.getvalue())
        ref_mu = np.mean(real_feats, axis=0)
        ref_sigma = np.cov(real_feats, rowvar=False)
        with tf.io.gfile.GFile(
                os.path.join(fdir, f'features/{config.data.dataset}_stats.npz'), "wb") as fout:
            io_buffer = io.BytesIO()
            np.savez_compressed(io_buffer, mu=ref_mu, sigma=ref_sigma)
            fout.write(io_buffer.getvalue())
    else:
        stats = np.load(os.path.join(fdir, f'features/{config.data.dataset}_stats.npz'))
        ref_mu = stats['mu']
        ref_sigma = stats['sigma']
    return ref_mu, ref_sigma

"""
Compute the FID stats from a generator model
"""
def get_model_features(G, model, mode="clean", z_dim=512, 
        num_gen=50_000, batch_size=128,
        device=torch.device("cuda"), desc="FID model: "):
    fn_resize = build_resizer(mode)
    # Generate test features
    num_iters = int(np.ceil(num_gen / batch_size))
    l_feats = []
    for idx in tqdm(range(num_iters), desc=desc):
        with torch.no_grad():
            z_batch = torch.randn((batch_size, z_dim)).to(device)
            # generated image is in range [0,255]
            img_batch = G(z_batch)
            # split into individual batches for resizing if needed
            if mode != "legacy_tensorflow":
                resized_batch = torch.zeros(batch_size, 3, 299, 299)
                for idx in range(batch_size):
                    curr_img = img_batch[idx]
                    img_np = curr_img.cpu().numpy().transpose((1, 2, 0))
                    img_resize = fn_resize(img_np)
                    resized_batch[idx] = torch.tensor(img_resize.transpose((2, 0, 1)))
            else:
                resized_batch = img_batch
            feat = get_batch_features(resized_batch, model, device)
        l_feats.append(feat)
    np_feats = np.concatenate(l_feats)
    return np_feats


"""
Computes the FID score for a generator model for a specific dataset 
and a specific resolution
"""
def fid_model(G, dataset_name, dataset_res, dataset_split,
              model=None, z_dim=512, num_gen=50_000,
              mode="clean", num_workers=0, batch_size=128,
              device=torch.device("cuda")):
    # define the model if it is not specified
    if model is None:
        model = build_feature_extractor(mode, device)
    # Load reference FID statistics (download if needed)
    ref_mu, ref_sigma = get_reference_statistics(dataset_name, dataset_res,
                                                 mode=mode, seed=0, split=dataset_split)
    # build resizing function based on options
    fn_resize = build_resizer(mode)

    # Generate test features
    np_feats = get_model_features(G, model, mode=mode,
        z_dim=z_dim, num_gen=num_gen,
        batch_size=batch_size, device=device)

    mu = np.mean(np_feats, axis=0)
    sigma = np.cov(np_feats, rowvar=False)
    fid = frechet_distance(mu, sigma, ref_mu, ref_sigma)
    return fid


"""
Computes the FID score between the two given folders
"""
def compare_folders(fdir1, fdir2, feat_model, mode, num_workers=0,
                    batch_size=8, device=torch.device("cuda")):
    # get all inception features for the first folder
    fbname1 = os.path.basename(fdir1)
    np_feats1 = get_folder_features(fdir1, feat_model, num_workers=num_workers,
                                    batch_size=batch_size, device=device, mode=mode, 
                                    description=f"FID {fbname1} : ")
    mu1 = np.mean(np_feats1, axis=0)
    sigma1 = np.cov(np_feats1, rowvar=False)
    # get all inception features for the second folder
    fbname2 = os.path.basename(fdir2)
    np_feats2 = get_folder_features(fdir2, feat_model, num_workers=num_workers,
                                    batch_size=batch_size, device=device, mode=mode,
                                    description=f"FID {fbname2} : ")
    mu2 = np.mean(np_feats2, axis=0)
    sigma2 = np.cov(np_feats2, rowvar=False)
    fid = frechet_distance(mu1, sigma1, mu2, sigma2)
    return fid


"""
Test if a custom statistic exists
"""
def test_stats_exists(name, mode):
    stats_folder = os.path.join(os.path.dirname(cleanfid.__file__), "stats")
    split, res="custom", "na"
    fname = f"{name}_{mode}_{split}_{res}.npz"
    fpath = os.path.join(stats_folder, fname)
    return os.path.exists(fpath)

def remove_custom_stats(name, mode="clean"):
    stats_folder = os.path.join(os.path.dirname(cleanfid.__file__), "stats")
    split, res="custom", "na"
    outname = f"{name}_{mode}_{split}_{res}.npz"
    outf = os.path.join(stats_folder, outname)
    if not os.path.exists(outf):
        msg = f"The stats file {name} does not exist."
        raise Exception(msg)
    os.remove(outf)


"""
Cache a custom dataset statistics file
"""
def make_custom_stats(name, fdir, num=None, mode="clean", 
                    num_workers=0, batch_size=64, device=torch.device("cuda")):
    stats_folder = os.path.join(os.path.dirname(cleanfid.__file__), "stats")
    split, res = "custom", "na"
    outname = f"{name}_{mode}_{split}_{res}.npz"
    outf = os.path.join(stats_folder, outname)
    # if the custom stat file already exists
    if os.path.exists(outf):
        msg = f"The statistics file {name} already exists. "
        msg += f"Use remove_custom_stats function to delete it first."
        raise Exception(msg)

    feat_model = build_feature_extractor(mode, device)
    fbname = os.path.basename(fdir)
    # get all inception features for folder images
    np_feats = get_folder_features(fdir, feat_model, num_workers=num_workers, num=num,
                                    batch_size=batch_size, device=device,
                                    mode=mode, description=f"FID {fbname} : ")
    mu = np.mean(np_feats, axis=0)
    sigma = np.cov(np_feats, rowvar=False)
    print(f"saving custom stats to {outf}")
    np.savez_compressed(outf, mu=mu, sigma=sigma)


def compute_kid(fdir1=None, fdir2=None, gen=None, 
            mode="clean", num_workers=12, batch_size=32,
            device=torch.device("cuda"), dataset_name="FFHQ",
            dataset_res=1024, dataset_split="train", num_gen=50_000, z_dim=512):
    # build the feature extractor based on the mode
    feat_model = build_feature_extractor(mode, device)
    
    # if both dirs are specified, compute FID between folders
    if fdir1 is not None and fdir2 is not None:
        print("compute KID between two folders")
        # get all inception features for the first folder
        fbname1 = os.path.basename(fdir1)
        np_feats1 = get_folder_features(fdir1, None, num_workers=num_workers,
                            batch_size=batch_size, device=device, mode=mode, 
                            description=f"KID {fbname1} : ")
        # get all inception features for the second folder
        fbname2 = os.path.basename(fdir2)
        np_feats2 = get_folder_features(fdir2, None, num_workers=num_workers,
                            batch_size=batch_size, device=device, mode=mode, 
                            description=f"KID {fbname2} : ")
        score = kernel_distance(np_feats1, np_feats2)
        return score

    # compute fid of a folder
    elif fdir1 is not None and fdir2 is None:
        print(f"compute KID of a folder with {dataset_name} statistics")
        # define the model if it is not specified
        model = build_feature_extractor(mode, device)
        ref_feats = get_reference_statistics(dataset_name, dataset_res,
                            mode=mode, seed=0, split=dataset_split, metric="KID")
        fbname = os.path.basename(fdir)
        # get all inception features for folder images
        np_feats = get_folder_features(fdir, model, num_workers=num_workers,
                                        batch_size=batch_size, device=device,
                                        mode=mode, description=f"KID {fbname} : ")
        score = kernel_distance(ref_feats, np_feats)
        return score

    # compute fid for a generator
    elif gen is not None:
        print(f"compute KID of a model with {dataset_name}-{dataset_res} statistics")
        # define the model if it is not specified
        model = build_feature_extractor(mode, device)
        ref_feats = get_reference_statistics(dataset_name, dataset_res,
                            mode=mode, seed=0, split=dataset_split, metric="KID")
        # build resizing function based on options
        fn_resize = build_resizer(mode)
        # Generate test features
        np_feats = get_model_features(gen, model, mode=mode,
            z_dim=z_dim, num_gen=num_gen, desc="KID model: ",
            batch_size=batch_size, device=device)
        score = kernel_distance(ref_feats, np_feats)
        return score
    
    else:
        raise ValueError(f"invalid combination of directories and models entered")


def compute_fid(config=None, fdir1=None, fdir2=None, gen=None, dataset=None, sigma_min=1e-3,
            mode="clean", num_workers=12, batch_size=32,
            device=torch.device("cuda"), dataset_name="FFHQ", assetdir='',
            dataset_res=256, dataset_split="train", num_gen=50_000, z_dim=512, dequantization=True,
                num_data=1000):
    # build the feature extractor based on the mode
    feat_model = build_feature_extractor(mode, device)

    # if both dirs are specified, compute FID between folders
    if fdir1 is not None and fdir2 is not None:
        print("compute FID between two folders")
        score = compare_folders(fdir1, fdir2, feat_model,
            mode=mode, batch_size=batch_size,
            num_workers=num_workers, device=device)
        return score

    # compute fid of a folder
    elif fdir1 is not None and fdir2 is None:
        print(f"compute FID of a folder with {dataset_name} statistics")
        scores = {}
        score = fid_folder(fdir1, dataset, dataset_name, dataset_res, dataset_split,
            model=feat_model, mode=mode, num_workers=num_workers,
            batch_size=batch_size, device=device, assetdir=assetdir, config=config, dequantization=True,
                           num_data=num_data)
        scores[0] = score
        return scores

    # compute fid for a generator
    elif gen is not None:
        print(f"compute FID of a model with {dataset_name}-{dataset_res} statistics")
        score = fid_model(gen, dataset_name, dataset_res, dataset_split,
                model=feat_model, z_dim=z_dim, num_gen=num_gen,
                mode=mode, num_workers=num_workers, batch_size=batch_size,
                device=device)
        return score
    
    else:
        raise ValueError(f"invalid combination of directories and models entered")
