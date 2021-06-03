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
    Stable version by Dougal J. Sutherland (not Author).
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
    with torch.no_grad():
        feat = model(batch.to(device))
    return feat.detach().cpu().numpy()

"""
Compute the inception features for a list of files
"""
def get_files_features(fdir, l_files, model=None, num_workers=12,
                       batch_size=128, device=torch.device("cuda"),
                       mode="clean", custom_fn_resize=None, 
                       description="", sampling_itr=''):
    # define the model if it is not specified
    if model is None:
        model = build_feature_extractor(mode, device)
    
    # build resizing function based on options
    if custom_fn_resize is not None:
        fn_resize = custom_fn_resize
    else:
        fn_resize = build_resizer(mode)
    
    '''# wrap the images in a dataloader for parallelizing the resize operation
    dataset = ResizeDataset(l_files, fn_resize=fn_resize)
    dataloader = torch.utils.data.DataLoader(dataset,
                    batch_size=batch_size, shuffle=False,
                    drop_last=False, num_workers=num_workers)

    # collect all inception features
    l_feats = []
    for batch in tqdm(dataloader, desc=description):
        l_feats.append(get_batch_features(batch, model, device))
    np_feats = np.concatenate(l_feats)'''

    l_feats = []
    transforms = torchvision.transforms.ToTensor()
    file_itr = 0
    for file in l_files:
        if sampling_itr == 0:
            save_filename = f'np_feats_{mode}_{file.split("/")[-1].split(".")[0].split("_")[1]}.npz'
        else:
            save_filename = f'np_feats_{mode}_{file.split("/")[-1].split(".")[0].split("_")[1]}_{file.split("/")[-1].split(".")[0].split("_")[3]}.npz'
        if not os.path.exists(os.path.join(fdir, f'features/{save_filename}')):
            img_np = np.load(file)['samples'].reshape(-1,256,256,3)
            itr = 0
            for img in img_np:
                img_resized = fn_resize(img)
                if img_resized.dtype == "uint8":
                    img_t = transforms(np.array(img_resized)) * 255
                elif img_resized.dtype == "float32":
                    img_t = transforms(img_resized)
                if itr == 0:
                    batch = img_t.reshape((1,)+img_t.shape)
                else:
                    batch = torch.cat((batch, img_t.reshape((1,) + img_t.shape)))
                if itr == 0:
                    print("batch: ", torch.min(batch[0]), torch.max(batch[0]))
                    print("--------------------------")
                    print(f"file iter : {file_itr}/{len(l_files)}")
                    file_itr += 1
                itr += 1
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
        print(f"l_feats stats : {np.min(l_feats[-1]), np.max(l_feats[-1])}")
        print("l_feats size : ", len(l_feats))
    np_feats = np.concatenate(l_feats)
    #print("HoHo : ", l_feats[0][0][:10])
    return np_feats

"""
Compute the inception features for a folder of features
"""
def get_folder_features(fdir, model=None, num_workers=12, num=None,
                        shuffle=False, seed=0, batch_size=128, device=torch.device("cuda"),
                        mode="clean", custom_fn_resize=None, description="", sampling_itr=0):
    # get all relevant files in the dataset
    files = sorted([file for ext in EXTENSIONS
                    for file in glob(os.path.join(fdir, f"*.{ext}"))])
    files_tmp = []
    #print("file directory : ", fdir)
    #print("files : ", files)
    for file in files:
        print(file)
        print(file.split('/')[-1])
        if sampling_itr == 0:
            if len(file.split('/')[-1].split('_')) == 2:
                files_tmp.append(file)

        else:
            if len(file.split('/')[-1].split('_')) > 2:
                if int(file.split('/')[-1].split('_')[3]) == sampling_itr:
                    files_tmp.append(file)

    files = files_tmp
    #print("files: ", files)
    if num is not None:
        if shuffle:
            random.seed(seed)
            random.shuffle(files)
        files = files[:num]
    np_feats = get_files_features(fdir, files, model, num_workers=num_workers,
                                  batch_size=batch_size, device=device,
                                  mode=mode,
                                  custom_fn_resize=custom_fn_resize,
                                  description=description,
                                  sampling_itr=sampling_itr)
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
               batch_size=128, device=torch.device("cuda"),
               sampling_itr=0, assetdir='', config=None):
    # define the model if it is not specified
    if model is None:
        model = build_feature_extractor(mode, device)
    # Load reference FID statistics (download if needed)
    if dataset == []:
        ref_mu, ref_sigma, stats = get_statistics(config, assetdir, mode)
    else:
        ref_mu, ref_sigma = get_statistics_from_dataset(fdir, config, dataset, mode, device)
    #ref_mu, ref_sigma = get_reference_statistics(dataset_name, dataset_res,
    #                                mode=mode, seed=0, split=dataset_split)
    fbname = os.path.basename(fdir)
    # get all inception features for folder images
    if not os.path.exists(os.path.join(fdir, 'features')):
        os.mkdir(os.path.join(fdir, 'features'))
    if os.path.exists(os.path.join(fdir, f'features/np_feats_{mode}_{sampling_itr}.npz')):
        np_feats = np.load(os.path.join(fdir, f'features/np_feats_{mode}_{sampling_itr}.npz'))['np_feats']
    else:
        np_feats = get_folder_features(fdir, model, num_workers=num_workers,
                                       batch_size=batch_size, device=device,
                                       mode=mode, description=f"FID {fbname} : ", sampling_itr=sampling_itr)
        with tf.io.gfile.GFile(
                os.path.join(fdir, f'features/np_feats_{mode}_{sampling_itr}.npz'), "wb") as fout:
            io_buffer = io.BytesIO()
            np.savez_compressed(io_buffer, np_feats=np_feats)
            fout.write(io_buffer.getvalue())
    if dataset_name == 'FFHQ':
        num_samples = 70000
    else:
        num_samples = 50000
    np_feats = np_feats[:num_samples]
    num_samples = np_feats.shape[0]
    logging.info(f'Number of samples : {num_samples}')
    print("shape of np_feats : ", np_feats.shape)
    #print("!! : ", np_feats[0][:10])
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
    if 'mu' in list(stats.keys()):
        return stats["mu"], stats["sigma"], []
    else:
        stats = stats['real_feats']
        return np.mean(stats, axis=0), np.cov(stats, rowvar=False), stats

def get_statistics_from_dataset(fdir, config, dataset, mode, device):
    if not os.path.exists(os.path.join(fdir, 'features/real_feats.npz')):
        model = build_feature_extractor(mode, device)
        scaler = datasets.get_data_scaler(config)
        # build resizing function based on options
        fn_resize = build_resizer(mode)
        transforms = torchvision.transforms.ToTensor()

        real_feats = []
        #for batch_id in range(len(dataset)):
        #for i, imgs in enumerate(dataset):
        data_num = 70000
        batch_size = 200
        for i in range(data_num // batch_size):
            imgs_np = next(dataset)['image']._numpy()
            itr = 0
            for img in imgs_np:
                img_resized = fn_resize(img)
                if img_resized.dtype == "uint8":
                    img_t = transforms(np.array(img_resized)) * 255
                elif img_resized.dtype == "float32":
                    img_t = transforms(img_resized)
                if itr == 0:
                    batch = img_t.reshape((1,) + img_t.shape)
                else:
                    batch = np.concatenate((batch, img_t.reshape((1,) + img_t.shape)))
                itr += 1
            #print("batch stats : ", np.min(batch), np.max(batch))
            print("--------------------------")
            print(f"real dataset : {i * imgs_np.shape[0]}")
            real_feats.append(get_batch_features(torch.tensor(batch) * 255., model, device))
            print(f"batch stats : {np.min(real_feats[-1]), np.max(real_feats[-1])}")
            tf.io.gfile.makedirs(os.path.join(fdir, 'features/'))
            with tf.io.gfile.GFile(
                    os.path.join(fdir, f'features/real_feats_{mode}_{i}.npz'), "wb") as fout:
                io_buffer = io.BytesIO()
                np.savez_compressed(io_buffer, real_feats=np.array(real_feats[-1]))
                fout.write(io_buffer.getvalue())
        real_feats = np.concatenate(real_feats, axis=0)
        tf.io.gfile.makedirs(os.path.join(fdir, 'features/'))
        with tf.io.gfile.GFile(
                os.path.join(fdir, f'features/real_feats_{mode}.npz'), "wb") as fout:
            io_buffer = io.BytesIO()
            np.savez_compressed(io_buffer, real_feats=real_feats)
            fout.write(io_buffer.getvalue())
    else:
        real_feats = np.load(os.path.join(fdir, f'features/real_feats.npz'))['real_feats']
    ref_mu = np.mean(real_feats, axis=0)
    ref_sigma = np.cov(real_feats, rowvar=False)
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
            dataset_res=256, dataset_split="train", num_gen=50_000, z_dim=512):
    # build the feature extractor based on the mode
    feat_model = build_feature_extractor(mode, device)

    if sigma_min == 1e-3:
        print("I am working with {}".format(1e-3))
        sampling_itrs = [0, 1280, 1640]
    elif sigma_min == 1e-4:
        print("I am working with {}".format(1e-4))
        sampling_itrs = [0, 1140, 1420, 1700]
    #sampling_itrs = [0]
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
        for sampling_itr in sampling_itrs:
            score = fid_folder(fdir1, dataset, dataset_name, dataset_res, dataset_split,
                model=feat_model, mode=mode, num_workers=num_workers,
                batch_size=batch_size, device=device, sampling_itr=sampling_itr, assetdir=assetdir, config=config)
            scores[sampling_itr] = score
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
