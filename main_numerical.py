# Numerical experiments for theory validation

import argparse
import csv
import json
import os
import random
from itertools import chain
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.stats import wishart
from sklearn import kernel_ridge, linear_model
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

import encoders
import invertible_network_utils
import latent_spaces
import losses
import spaces
import utils

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

print("device:", device)


# ---------------------------- parser --------------------------
# ---------------------------------------------------------------
def parse_args():
    """
    Parse command line arguments.

    Returns:
        args (argparse.Namespace): Parsed arguments.
        parser (argparse.ArgumentParser): Argument parser object.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, default="results/numerical")
    parser.add_argument("--model-id", type=str, default="gumbel_softmax")
    parser.add_argument("--latent-dim", type=int, default=6)
    parser.add_argument("--encoding-size", type=int, default=20)
    parser.add_argument("--evaluate", action="store_true")  # by default false
    parser.add_argument("--num-train-batches", type=int, default=5)
    parser.add_argument("--num-eval-batches", type=int, default=10)
    parser.add_argument("--seed", type=int, default=np.random.randint(32**2 - 1))
    parser.add_argument("--n-mixing-layer", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)  # 1e-4 for hard
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--n-log-steps", type=int, default=100)
    parser.add_argument("--n-steps", type=int, default=100001)
    parser.add_argument("--resume-training", action="store_false")
    parser.add_argument("--n-views", type=int, default=4)  # number of views we consider
    parser.add_argument(
        "--S-k",
        type=int,
        help="view-specific latents",
        default=[[0, 1, 2, 3, 4], [0, 1, 2, 4, 5], [0, 1, 2, 3, 5], [0, 1, 3, 4, 5]],
    )
    parser.add_argument("--grid-search-eval", action="store_true")
    parser.add_argument("--shared-mixing-function", type=bool, default=False)
    parser.add_argument("--shared-encoder", type=bool, default=False)
    parser.add_argument(
        "--selection",
        type=str,
        default="ground_truth",
        choices=["ground_truth", "gumbel_softmax", "concat"],
    )
    parser.add_argument("--subsets", default=None)
    parser.add_argument("--evaluate_individual_latents", action="store_true")
    parser.add_argument("--n_dependent_dims", default=0, type=int)
    args = parser.parse_args()
    return args, parser


# ------ store content and style dict into args for global use ---------
# --------------------------------------------------------------------------
def update_args(args):
    """
    Update the arguments with view-specific latents, subsets, content dictionary, style dictionary,
    content size dictionary, latent dimension, and view-specific content indexing based on the selection.

    Args:
        args (Namespace): The input arguments.

    Returns:
        Namespace: The updated arguments.
    """
    zs_views = torch.tensor(args.S_k)  # [n_views, n_sk] # the view-specific latents as given in args.

    # retrieve subsets, content dict and style dict for all subsets and views
    if args.subsets is None:
        powerset, _ = utils.powerset(range(args.n_views), only_consider_whole_set=False)
        setattr(args, "subsets", powerset)

    content_dict, style_dict = utils.content_style_from_subsets(subsets=args.subsets, zs=zs_views)
    setattr(args, "content_dict", content_dict)
    setattr(args, "style_dict", style_dict)

    # store content size, for the mode: known content size
    content_size_dict = {}
    for k, v in content_dict.items():
        content_size_dict[k] = len(v)
    args.content_size_dict = content_size_dict

    # make sure the number of latents align with Sk
    zn_set = list(set(chain.from_iterable(args.S_k)))
    args.latent_dim = len(zn_set)

    view_specific_content_indexing = {s: {} for s in args.subsets}
    if args.selection == "ground_truth":
        for s in args.subsets:
            for k in s:
                view_specific_content_indexing[s][k] = [args.S_k[k].index(c) for c in args.content_dict[s]]
        args.view_specific_content_indexing = view_specific_content_indexing
    elif args.selection == "concat":  # concat all content indices
        est_content_indices = np.array_split(range(args.encoding_size), len(args.subsets))
        args.est_content_dict = {
            subset: {k: indices for k in subset} for subset, indices in zip(args.subsets, est_content_indices)
        }
    return args


def load_config_dict():
    """
    Load the configuration dictionary from the 'fzoo.yaml' file and return the solver and model configurations.

    Returns:
        Tuple[utils.ConfigDict, utils.ConfigDict]: A tuple containing the solver and model configurations.
    """
    config_dict = yaml.safe_load(Path("configs/fzoo.yaml").read_text())
    config_solver = utils.ConfigDict(config_dict["solver"])
    config_model = utils.ConfigDict(config_dict["model"])
    return config_solver, config_model


# ---------- initialisation functions ----------------------
# ----------------------------------------------------------
def init_or_load_mixing_fns(device, args):
    """
    Initializes or loads the mixing functions for the multi-view case.

    Args:
        device (torch.device): The device to use for computation.
        args (argparse.Namespace): The command-line arguments.

    Returns:
        torch.nn.ModuleList: The list of mixing functions.
    """
    # Invertible MLP requires the same input and the same output size
    # extend to multi-view case
    F = torch.nn.ModuleList()  # set of mixing functions, not trainable after generated.
    for i in range(args.n_views):
        f_i = invertible_network_utils.construct_invertible_mlp(
            n=len(args.S_k[i]),
            n_layers=args.n_mixing_layer,
            cond_thresh_ratio=0.001,
            n_iter_cond_thresh=25000,
        )
        F.append(f_i)
    if args.evaluate:
        F = torch.nn.ModuleList()
        mixing_fn_state_dict = torch.load(os.path.join(args.save_dir, "mixing_fns.pt"))
        for i, param_dict in mixing_fn_state_dict.items():
            f_i = invertible_network_utils.construct_invertible_mlp(
                n=len(args.S_k[i]),
                n_layers=args.n_mixing_layer,
                cond_thresh_ratio=0.001,
                n_iter_cond_thresh=25000,
            )
            f_i.load_state_dict(param_dict)
            f_i.to(device)
            F.append(f_i)
            # disable gradient descent for mixing functions
            for p in f_i.parameters():
                p.requires_grad = False

    if args.shared_mixing_function:
        F = [F[0]] * args.n_views
    return F


def init_or_load_encoder_models(device, args, encoding_size=None):
    """
    Initialize or load encoder models.

    Args:
        device (torch.device): The device to use for the models.
        args (argparse.Namespace): The command-line arguments.
        encoding_size (int, optional): The size of the encoding. Defaults to None.

    Returns:
        torch.nn.ModuleList: A list of encoder models.
    """
    G = torch.nn.ModuleList()
    for i in range(args.n_views):
        g_i = encoders.get_mlp(
            n_in=len(args.S_k[i]),
            n_out=encoding_size or len(args.S_k[i]),
            layers=[
                len(args.S_k[i]) * 10,
                len(args.S_k[i]) * 50,
                len(args.S_k[i]) * 50,
                len(args.S_k[i]) * 50,
                len(args.S_k[i]) * 50,
                len(args.S_k[i]) * 10,
            ],
        )
        G.append(g_i)
        g_i.to(device)
    if args.evaluate:
        G = torch.nn.ModuleList()

        save_path = os.path.join(args.save_dir, "model.pt")
        ckpt = torch.load(save_path)

        for i in range(args.n_views):
            g_i = encoders.get_mlp(
                n_in=len(args.S_k[i]),
                n_out=encoding_size or len(args.S_k[i]),
                layers=[
                    len(args.S_k[i]) * 10,
                    len(args.S_k[i]) * 50,
                    len(args.S_k[i]) * 50,
                    len(args.S_k[i]) * 50,
                    len(args.S_k[i]) * 50,
                    len(args.S_k[i]) * 10,
                ],
            )
            g_i.load_state_dict(ckpt[f"encoder_{i}_state_dict"])
            g_i.to(device)
            G.append(g_i)
    if args.shared_encoder:
        G = [G[0]] * args.n_views
    return G


def init_or_load_training_models(mixing_fns, encoderes, device, args):
    """
    Initialize or load the training models.

    Args:
        mixing_fns (list): List of mixing functions.
        encoderes (list): List of encoders.
        device (torch.device): The device to use for computation.
        args: Additional arguments.

    Returns:
        dict: A dictionary containing the initialized or loaded models.
    """

    # torch.nn.Module wrapper for encoder-mixing_function composition
    backbone = encoders.CompositionEncMix(mixing_fns=mixing_fns, encoders=encoderes)
    backbone.to(device)

    return {"backbone": backbone}


def init_or_load_optimizer(models: dict, optimizer_class=torch.optim.Adam, args=None):
    """
    Initialize or load the optimizer for the models.

    Args:
        models (dict): A dictionary containing the models.
        optimizer_class (torch.optim.Optimizer): The optimizer class to use (default: torch.optim.Adam).
        args (argparse.Namespace): The command-line arguments (default: None).

    Returns:
        tuple: A tuple containing the trainable parameters and the optimizer.
    """
    # initialise trainable parameters
    params = []
    if args.shared_encoder:
        params = models["backbone"].encoders[0].parameters()
    else:
        for g_i in models["backbone"].encoders:
            params = params + list(g_i.parameters())  # encoders' parameters are trainable

    """Define Adam optimiser"""
    optimizer = optimizer_class(params, lr=args.lr)
    return params, optimizer


# ---------------- checkpoint and resume training ---------------
# -----------------------------------------------------------------------
def save_mixing_fns(args, mixing_fns):
    """
    Save the state dictionaries of the mixing functions to a file.

    Args:
        args (Namespace): Command-line arguments.
        mixing_fns (list): List of mixing functions.

    Returns:
        None
    """
    n_views = len(mixing_fns)
    state_dict = {}
    for i in range(n_views):
        state_dict[i] = mixing_fns[i].state_dict()

    save_path = os.path.join(args.save_dir, "mixing_fns.pt")
    torch.save(state_dict, save_path)


def save_models(args, models: dict, optimizer=None):
    """
    Save the models and optimizer state_dict to a file.

    Args:
        args (Namespace): The command line arguments.
        models (dict): A dictionary containing the models.
        optimizer (Optimizer, optional): The optimizer. Defaults to None.
    """
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    state_dict = {}

    for k in range(args.n_views):
        state_dict[f"encoder_{k}_state_dict"] = models["backbone"].encoders[k].state_dict()

    if optimizer is not None:
        state_dict["optimizer_state_dict"] = optimizer.state_dict()

    save_path = os.path.join(args.save_dir, "model.pt")
    torch.save(state_dict, save_path)


def load_models(models, optimizer, args):
    """
    Load models and optimizer from a saved checkpoint.

    Args:
        models (dict): A dictionary containing the models.
        optimizer (torch.optim.Optimizer): The optimizer.
        args (argparse.Namespace): The command-line arguments.

    Returns:
        tuple: A tuple containing the loaded models and optimizer.
    """
    save_path = os.path.join(args.save_dir, "model.pt")
    ckpt = torch.load(save_path)

    for k in range(args.n_views):
        models["backbone"].encoders[k].load_state_dict(ckpt[f"encoder_{k}_state_dict"])

    if "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return models, optimizer


def infer_content_indices_gumbel_softmax(args, hzs: dict, content_size_dict: dict):
    """
    Infer content indices using Gumbel Softmax (content sizes predefined).

    Args:
        args: Arguments for the function.
        hzs (dict): Dictionary containing the hz values.
        content_size_dict (dict): Dictionary containing the content size for each subset.

    Returns:
        dict: Dictionary containing the estimated content indices for each subset and view.
    """
    est_content_dict = {subset: {} for subset in args.subsets}
    for subset in args.subsets:
        for k in subset:
            avg_logits = hzs[k]["hz"].mean(0)[None]
            m = utils.topk_gumbel_softmax(
                k=content_size_dict[subset],
                logits=avg_logits,  # hzs[k]["hz"][0][None],
                tau=1.0,
                hard=True,
            )
            c_ind = torch.where(m)[-1].tolist()  # batch_size, nSk
            est_content_dict[subset][k] = c_ind  # this indicies is different for different views
    return est_content_dict


def infer_content_indices(args, hzs, *mode_specific_args):
    """
    Infer the content indices based on the given arguments and mode-specific arguments.

    Args:
        args: The arguments object containing the selection mode.
        hzs: The hzs object.
        mode_specific_args: Additional arguments specific to the selected mode.

    Returns:
        The inferred content indices.

    Raises:
        ValueError: If the selection mode is not supported.
    """
    if args.selection == "ground_truth":
        return args.view_specific_content_indexing
    elif args.selection == "concat":
        return args.est_content_dict
    elif args.selection == "gumbel_softmax":
        return infer_content_indices_gumbel_softmax(args, hzs, *mode_specific_args)
    else:
        raise ValueError(f"mode={args.selection} not supported")


# ----------------- data generation ----------------------
# -----------------------------------------------------------------
def sample_whole_latent(latent_space, size, device=device):
    """
    Samples latent vectors from the given latent space.

    Args:
        latent_space (LatentSpace): The latent space object.
        size (int): The number of latent vectors to sample.
        device (torch.device, optional): The device to use for sampling. Defaults to the global device.

    Returns:
        tuple: A tuple containing two tensors - the positive sample and the negative sample.
    """
    z = latent_space.sample_latent(size=size, device=device)  # positive sample
    z3 = latent_space.sample_latent(size=size, device=device)  # negative sample
    return z, z3


def generate_data(latent_space, models, num_batches=1, batch_size=4096, loss_func=None, args=None):
    """
    Generate data for training or evaluation.

    Args:
        latent_space (LatentSpace): The latent space object used for sampling latent vectors.
        models (dict): A dictionary of models, including the backbone model.
        num_batches (int, optional): The number of batches to generate. Defaults to 1.
        batch_size (int, optional): The batch size. Defaults to 4096.
        loss_func (callable, optional): The loss function to use. Defaults to None.
        args (argparse.Namespace, optional): Additional arguments. Defaults to None.

    Returns:
        tuple: A tuple containing the data dictionary, hz dictionary, and all_z tensor.
            - data_dict (dict): A dictionary containing the generated data for each subset and view.
            - hz_dict (dict): A dictionary containing the computed hz values for each view and subset.
            - all_z (numpy.ndarray): A numpy array containing all the sampled latent vectors.

    """
    models["backbone"].eval()

    data_dict = {subset: {k: {"c": [], "s": []} for k in subset} for subset in args.subsets}

    hz_dict = {
        k: {
            "hz": [],  # unified encoded information
            "est_c_ind": {s: [] for s in args.subsets if k in s},  # for all subsets
        }
        for k in range(args.n_views)
    }

    all_z = []

    with torch.no_grad():
        for _ in range(num_batches):
            zs = latent_space.sample_latent(batch_size)  # [batch_size, n_z]
            all_z += [zs]

            hzs = dict({})

            # compute the estimated latents for each view (using the unified encoder)
            for k in range(args.n_views):
                hz = models["backbone"].view_specific_forward(zs, k, args.S_k)  # [batch_size, nz]
                hzs[k] = {"hz": hz}  # to compute the readout, preserve ternsor type
                hz_dict[k]["hz"].append(hz.detach().cpu().numpy())

            for subset_idx, subset in enumerate(args.subsets):
                content_z = zs[:, list(args.content_dict[subset])]
                for k_idx, k in enumerate(subset):
                    style_z = zs[:, list(args.style_dict[subset][k])]
                    # z_Sk = zs[:, args.S_k[k]]

                    est_content_dict = infer_content_indices(args, hzs, args.content_size_dict)
                    # append data
                    data_dict[subset][k]["c"].append(content_z.detach().cpu().numpy())
                    data_dict[subset][k]["s"].append(style_z.detach().cpu().numpy())

                    hz_dict[k]["est_c_ind"][subset].append(est_content_dict[subset][k])

        for subset, subset_dict in data_dict.items():
            for k, k_dict in subset_dict.items():
                data_dict[subset][k]["c"] = np.stack(k_dict["c"], axis=0)
                data_dict[subset][k]["s"] = np.stack(k_dict["s"], axis=0)

        for k, v in hz_dict.items():
            hz_dict[k]["hz"] = np.stack(v["hz"], axis=0)
            for subset in hz_dict[k]["est_c_ind"].keys():
                hz_dict[k]["est_c_ind"][subset] = np.stack(
                    v["est_c_ind"][subset], axis=0
                )  # [num_batches, batch_size, ...]

        return data_dict, hz_dict, torch.stack(all_z, 0).detach().cpu().numpy()


# ------------------ Training -----------------
# --------------------------------------------------------
def train_step(data, loss, models, optimizer, params, args, **kwargs):
    """
    Args:
        data = (z_positive, latent_dimegative),
        loss: loss class from losses.py
        H: {h = g circ f}_{k} with g being encoder and f predefined mixing function (for each view), shape [K, ],
        optimizer: optimizer object,
        params: parameter to optimize
    Returns:
        training loss
    """

    models["backbone"].train()
    # reset grad
    if optimizer is not None:
        optimizer.zero_grad()

    z, z3 = data
    z = z.to(device)
    z3 = z3.to(device)

    # forward pass
    z_rec, z3_rec, hzs = models["backbone"].forward(z=z, z3=z3, S_k=args.S_k, n_views=args.n_views)
    est_content_dict = infer_content_indices(args, hzs, args.content_size_dict)
    total_loss_value, _, _ = loss.loss(est_content_dict, z_rec, z3_rec)

    if optimizer is not None:
        total_loss_value.backward()
        optimizer.step()

    return total_loss_value.item()


def generate_latent_space(args):
    assert args.n_dependent_dims <= args.latent_dim
    latent_spaces_list = []
    Sigma_z_path = os.path.join(args.save_dir, "Sigma_z.csv")
    if not args.evaluate:
        if args.n_dependent_dims == 0:
            Sigma_z = np.eye(args.latent_dim)
        else:
            # In the non-dependent case, we generate a set of dependent and non-dependent latent variables
            Sigma_z = np.eye(args.latent_dim)
            Sigma_z_dep = wishart.rvs(args.n_dependent_dims, np.eye(args.n_dependent_dims), size=1)
            Sigma_z[: args.n_dependent_dims, : args.n_dependent_dims] = Sigma_z_dep

        np.savetxt(Sigma_z_path, Sigma_z, delimiter=",")
    else:
        Sigma_z = np.loadtxt(Sigma_z_path, delimiter=",")
        print(Sigma_z)
    space = spaces.NRealSpace(args.latent_dim)

    # Here just one latent space
    def sample_latent(space, size, device=device):
        return space.normal(None, 1.0, size, device, Sigma=Sigma_z)

    latent_spaces_list.append(latent_spaces.LatentSpace(space=space, sample_latent=sample_latent))
    latent_space = latent_spaces.ProductLatentSpace(spaces=latent_spaces_list)
    return latent_space


# ------------------ Evaluate --------------------
# ------------------------------------------------------------------


# Rest of the code...
def evaluate(models, latent_space, args):
    """
    Evaluate the performance of the models on the given latent space.

    Args:
        models (list): List of models to evaluate.
        latent_space (numpy.ndarray): Latent space data.
        args (object): Arguments object containing evaluation parameters.

    Returns:
        None
    """

    def generate_nonlinear_model():
        if not args.grid_search_eval:
            model = MLPRegressor(max_iter=5000)  # lightweight option
        else:
            # grid search is time- and memory-intensive
            model = GridSearchCV(
                kernel_ridge.KernelRidge(kernel="rbf", gamma=0.1),
                param_grid={
                    "alpha": [1e0, 0.1, 1e-2, 1e-3],
                    "gamma": np.logspace(-2, 2, 4),
                },
                cv=3,
                n_jobs=-1,
            )
        return model

    if args.evaluate:
        num_batches = args.num_eval_batches
        file_name = "Evaluation"
    else:
        num_batches = 1
        file_name = "Training"

    # lightweight evaluation with linear classifiers
    data_dict, hz_dict, all_zs = generate_data(
        latent_space=latent_space, models=models, num_batches=num_batches, args=args
    )

    # standardize the estimated latents hz
    data_shape = hz_dict[0]["hz"].shape  # [num_batches, batch_size, nSk]
    for k, v in hz_dict.items():
        hz_dict[k]["hz"] = StandardScaler().fit_transform(np.concatenate(v["hz"], axis=0)).reshape(*data_shape)

    # predict individual latents from the estimated content block
    for subset_idx, subset in enumerate(data_dict):
        scores = {latent_idx: {"linear": [], "nonlinear": []} for latent_idx in range(args.latent_dim)}
        for k in subset:
            for i in range(num_batches):
                predicted_content_idx = hz_dict[k]["est_c_ind"][subset][i]
                batch_size = hz_dict[k]["hz"][i].shape[0]
                inputs = np.take_along_axis(
                    hz_dict[k]["hz"][i], np.tile(predicted_content_idx[None], (batch_size, 1)), axis=-1
                )
                for latent_idx in range(args.latent_dim):
                    # labels = StandardScaler().fit_transform(data_dict[subset][k][keyword])
                    labels = all_zs[i, :, latent_idx][:, None]  # [batch_size, n_keyword]
                    (
                        train_inputs,
                        test_inputs,
                        train_labels,
                        test_labels,
                    ) = train_test_split(inputs, labels)
                    data = [train_inputs, train_labels, test_inputs, test_labels]
                    r2_linear = utils.evaluate_prediction(linear_model.LinearRegression(n_jobs=-1), r2_score, *data)
                    if args.evaluate:
                        # nonlinear regression
                        r2_nonlinear = utils.evaluate_prediction(generate_nonlinear_model(), r2_score, *data)
                    else:
                        r2_nonlinear = -1.0  # not computed
                    scores[latent_idx]["linear"].append(r2_linear)
                    scores[latent_idx]["nonlinear"].append(r2_nonlinear)
            for latent_idx in range(args.latent_dim):
                file_path = os.path.join(args.save_dir, f"{file_name}.csv")
                fileobj = open(file_path, "a+")
                writer = csv.writer(fileobj)
                wri = [
                    subset,
                    "view",
                    k,
                    "latent_idx",
                    latent_idx,
                    "linear mean",
                    f"{np.mean(scores[latent_idx]['linear']):.3f} +- {np.std(scores[latent_idx]['linear']) :.3f}",
                    "nonlinear mean",
                    f"{np.mean(scores[latent_idx]['nonlinear']):.3f} +- {np.std(scores[latent_idx]['nonlinear']):.3f}",
                ]
                writer.writerow(wri)
                fileobj.close()


# ------------------- main loop ------------------------------------
# ------------------------------------------
def main():
    args, parser = parse_args()
    args.save_dir = os.path.join(args.model_dir, args.model_id)
    os.makedirs(args.save_dir, exist_ok=True)
    with open(os.path.join(args.save_dir, "settings.json"), "w") as fp:
        json.dump(args.__dict__, fp, ensure_ascii=False)
    args = update_args(args)  # update subsetss and information

    if args.evaluate:
        args.n_steps = 1

    print("Arguments:")
    for k, v in vars(args).items():
        print(f"\t{k}: {v}")

    global device
    if args.no_cuda:
        device = "cpu"
        print("Using cpu")

    # set seed
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    """Initialisation"""
    mixing_fns = init_or_load_mixing_fns(device, args)  # mixing function always gives S_k
    encoders = init_or_load_encoder_models(
        device, args, encoding_size=args.encoding_size if args.selection == "concat" else None
    )
    models = init_or_load_training_models(mixing_fns=mixing_fns, encoderes=encoders, device=device, args=args)
    params, optimizer = init_or_load_optimizer(models=models, args=args)

    # initialise loss function
    loss = losses.UnifiedCLLoss(losses.LpSimCLRLoss())
    # initialise latent space
    latent_space = generate_latent_space(args)

    # save generative model / mixing_functions
    save_mixing_fns(args, mixing_fns)

    # ----------Training
    # --------------------------------------------
    if ("total_loss_values" in locals() and not args.resume_training) or "total_loss_values" not in locals():
        total_loss_values = []
        accs_global = []

    global_step = len(total_loss_values) + 1
    last_save_at_step = 0
    while global_step <= args.n_steps and not args.evaluate:
        data = sample_whole_latent(latent_space=latent_space, size=args.batch_size)
        total_loss_value = train_step(
            data=data,
            loss=loss,
            models=models,
            optimizer=optimizer,
            params=params,
            args=args,
        )

        # store losses
        total_loss_values.append(total_loss_value)

        # checkpoint & evaluate for every n_log_steps
        if global_step % args.n_log_steps == 1 or global_step == args.n_steps:
            save_models(args, models, optimizer)  # add step_idx for the models, otherwise will be overwrite
            evaluate(models, latent_space, args)
            print(
                f"Step: {global_step} \t",
                f"Loss: {total_loss_value:.4f} \t",
                f"<Loss>: {np.mean(np.array(total_loss_values[-args.n_log_steps:])):.4f} \t",
            )
        global_step += 1

    # ----- Evaluation
    # --------------------------------------
    if args.evaluate:
        evaluate(models, latent_space, args)


if __name__ == "__main__":
    main()
