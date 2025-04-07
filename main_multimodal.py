# Experiment with multimodal (image/text) data.

import argparse
import csv
import functools
import json
import operator
import os
import random
import uuid
import warnings

import faiss
import numpy as np
import pandas as pd
import torch
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import resnet18
from typing_extensions import Callable, List

import datasets
import dci
import utils
from encoders import TextEncoder2D
from infinite_iterator import InfiniteIterator
from losses import infonce_loss

device_ids = [0]


# ---------------------------- parser & args --------------------------
# ---------------------------------------------------------------
def parse_args():
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot", type=str, required=True)
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        default="multimpdal3di",
        choices=["mpi3d", "independent3di", "causal3di", "multimodal3di"],
    )
    parser.add_argument("--model-dir", type=str, default="results")
    parser.add_argument("--model-id", type=str, default="0")
    parser.add_argument("--encoding-size", type=int, default=32)
    parser.add_argument("--hidden-size", type=int, default=100)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--train-steps", type=int, default=300001)
    parser.add_argument("--log-steps", type=int, default=100)
    parser.add_argument("--checkpoint-steps", type=int, default=1000)
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--val-size", default=25000, type=int)
    parser.add_argument("--test-size", default=25000, type=int)
    parser.add_argument("--seed", type=int, default=np.random.randint(32**2 - 1))
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--no-cuda", action="store_true")
    parser.add_argument("--save-all-checkpoints", action="store_true")
    parser.add_argument("--resume-training", action="store_false")
    parser.add_argument("--load-args", action="store_true")
    parser.add_argument("--collate-random-pair", action="store_true")
    parser.add_argument("--modalities", default=["image"], choices=[["image"], ["image", "text"]])
    parser.add_argument(
        "--selection",
        type=str,
        default="concat",
        choices=["ground_truth", "gumbel_softmax", "concat", "soft"],
    )

    parser.add_argument("--n-views", default=3, type=int)
    parser.add_argument(
        "--change-lists", default=[[4, 5, 6, 8, 9, 10]]
    )  # list of latent indices we want to perturb in the augmented views
    parser.add_argument("--faiss-omp-threads", type=int, default=16)
    parser.add_argument("--subsets", default=[(0, 1), (0, 2), (1, 2), (0, 1, 2)])
    parser.add_argument("--eval-dci", action="store_true")
    parser.add_argument("--eval-style", action="store_true")
    parser.add_argument("--grid-search-eval", action="store_true")
    return parser


def update_args(args):
    """
    Update the initial arguments with computed subsets and corresponding latent style variables.

    Args:
        args (argparse.Namespace): The initial arguments.

    Returns:
        argparse.Namespace: The updated arguments.
    """
    if args.dataset_name == "independent3di":
        args.DATASETCLASS = datasets.Indepdenent3DIdent
        setattr(args, "modalities", ["image"])
    elif args.dataset_name == "causal3di":
        args.DATASETCLASS = datasets.Causal3DIdent
        setattr(args, "modalities", ["image"])
    elif args.dataset_name == "multimodal3di":
        args.DATASETCLASS = datasets.Multimodal3DIdent
        setattr(args, "modalities", ["image", "text"])
    elif args.dataset_name == "mpi3d":
        args.DATASETCLASS = datasets.MPI3D
        setattr(args, "modalities", ["image"])
        # only consider pair of views here, following locatello 2020
        assert args.n_views == 2, "mpi3d only consider pair of views: n-views=2"
        setattr(args, "n-views", 2)
        setattr(args, "subsets", [(0, 1)])
        setattr(args, "change_lists", [])
        setattr(args, "collate_random_pair", True)
    else:
        raise f"{args.dataset_name=} not supported."

    if len(args.subsets) == 1 or args.n_views == 2:  # Train content encoders
        setattr(args, "subsets", [tuple(range(args.n_views))])
        setattr(args, "content_indices", [list(range(args.encoding_size))])
    else:
        # Train view-specific encoders
        if not hasattr(args, "subsets"):
            subsets, _ = utils.powerset(range(args.n_views))  # compute the all subset of views which have >= 2 views
            setattr(args, "subsets", subsets)

        assert max(set().union(*args.subsets)) < args.n_views, "The given view is outside boundary!"

        if args.selection in ["ground_truth", "gumbel_softmax"]:
            # if require to compute GT content index, I have to have predefined changes and so on
            content_indices = compute_gt_idx(args)
            setattr(args, "content_indices", content_indices)
            setattr(args, "encoding_size", len(args.DATASETCLASS.FACTORS["image"]))
        elif args.selection == "concat":
            assert args.encoding_size > len(args.subsets)
            est_content_indices = np.array_split(range(args.encoding_size), len(args.subsets))
            setattr(args, "content_indices", [ind.tolist() for ind in est_content_indices])
        # compute independent indices
        content_union = set().union(*args.content_indices)
        style_indices = [i for i in range(args.encoding_size) if i not in content_union]
        setattr(args, "style_indices", style_indices)
    return args


# ------------------- compute content indices -------------------
# ---------------------------------------------------------------
def compute_gt_idx(args):
    """
    Compute the ground truth content indices based on the given arguments.

    Args:
        args: The arguments containing the dataset name and subsets.

    Returns:
        A list of ground truth content indices for each subset.
    """
    factors = args.DATASETCLASS.FACTORS["image"].keys()

    if args.dataset_name in ["independent3di", "causal3di"]:
        if args.dataset_name == "independent3di":
            setattr(args, "change_lists", [[4, 5, 6, 8, 9]])
        elif args.dataset_name == "causal3di":
            setattr(args, "change_lists", [[8, 9, 10], [1, 2, 3, 4, 5, 6, 7]])  # 1: change hues, 2: change pos and rot
        content_dict = {}
        indicators = [[True] * len(factors)]
        for view, change in enumerate(args.change_lists):
            indicators.append([z not in change for z in factors])
        # content_indices = [[0, 1, 2, 3, 4, 5, 6, 7], [0, 8, 9, 10], [0], [0]]
        for s in args.subsets:
            content_dict[s] = np.where(list(functools.reduce(operator.eq, [np.array(indicators[k]) for k in s])))[
                0
            ].tolist()
        return list(content_dict.values())

    elif args.dataset_name == "multimodal3di":
        # here, the last view is text
        # option 1
        setattr(args, "change_lists", [[1, 2, 3, 4, 5, 6, 7, 8, 9]])  # change rot + hues + spotlight pos
        content_dict = {}
        indicators = [[True] * len(factors)]
        for view, change in enumerate(args.change_lists):
            indicators.append([z not in change for z in factors])
        # indicator for text0
        indicators.append([True] * 3)
        for s in args.subsets:
            indicators_copy = indicators.copy()
            if 2 in s:  # treat text differently
                indicators_copy = [ind[: len(indicators[-1])] for ind in indicators]
            content_dict[s] = np.where(list(functools.reduce(operator.eq, [np.array(indicators_copy[k]) for k in s])))[
                0
            ].tolist()
        print(content_dict)
        return list(content_dict.values())
    else:
        raise f"No ground truth content computed for {args.dataset_name=} yet!"


def train_step(data, fs: List[Callable], loss_func, optimizer, params, args):
    """
    Perform a single training step.

    Args:
        data (dict): A dictionary containing the input data for each modality.
        fs (List[Callable]): A list of functions representing the feature extractors for each modality.
        loss_func (Callable): The loss function to compute the loss.
        optimizer (torch.optim.Optimizer): The optimizer to update the model parameters.
        params (Iterable[torch.Tensor]): The model parameters to be optimized.
        args (argparse.Namespace): Command-line arguments.

    Returns:
        tuple: A tuple containing the loss value and the estimated content indices.
    """
    # reset grad
    if optimizer is not None:
        optimizer.zero_grad()

    # compute loss
    hz = []  # concat the learned reprentation for all views
    n_views = int(0)
    for m_midx, m in enumerate(args.modalities):
        samples = data[m]
        hz_m = fs[m_midx](torch.concat(samples, 0))
        hz += [hz_m]
        n_views += len(samples)

    hz = torch.concat(
        hz,
        0,
    )

    avg_logits = hz.mean(0)[None]
    if "content_indices" not in data:
        data["content_indices"] = args.content_indices
    content_size = [len(content) for content in data["content_indices"]]  # (batch_size, )

    if args.selection in ["ground_truth", "concat"]:
        estimated_content_indices = args.content_indices  # len = len(subsets)
    else:
        if args.subsets[-1] == list(range(args.n_views)) and content_size[-1] > 0:
            # when the joint intersection is not empty,
            # we use the fact that the joint intersection will be in all smaller subsets
            content_masks = utils.smart_gumbel_softmax_mask(
                avg_logits=avg_logits, content_sizes=content_size, subsets=args.subsets
            )
        else:
            content_masks = utils.gumbel_softmax_mask(
                avg_logits=avg_logits, content_sizes=content_size, subsets=args.subsets
            )

        estimated_content_indices = []
        for c_mask in content_masks:
            c_ind = torch.where(c_mask)[-1].tolist()
            estimated_content_indices += [c_ind]

    loss_value = loss_func(hz.reshape(n_views, -1, hz.shape[-1]), estimated_content_indices, args.subsets)

    # backprop
    if optimizer is not None:
        loss_value.backward()
        clip_grad_norm_(params, max_norm=2.0, norm_type=2)  # stabilizes training
        optimizer.step()

    return loss_value.item(), estimated_content_indices


def val_step(data, fs, loss_func, args):
    """
    Perform a validation step.

    Args:
        data: The input data for the validation step.
        fs: The feature set for the validation step.
        loss_func: The loss function to be used.
        args: Additional arguments for the validation step.

    Returns:
        The result of the validation step.
    """
    return train_step(data, fs, loss_func, optimizer=None, params=None, args=args)


def get_data(dataset, fs, loss_func, dataloader_kwargs, num_samples=None, args=None):
    """
    Get data from the dataset and compute loss values and representations for each modality.

    Args:
        dataset (torch.utils.data.Dataset): The dataset to get data from.
        fs (list): List of functions to compute representations for each modality.
        loss_func: The loss function to compute the loss value.
        dataloader_kwargs (dict): Additional keyword arguments to pass to the DataLoader.
        num_samples (int, optional): The number of samples to process. If None, process all samples in the dataset.
        args (argparse.Namespace, optional): Additional arguments.

    Returns:
        dict: A dictionary containing the computed loss values and representations for each modality.
    """
    loader = DataLoader(dataset, **dataloader_kwargs)
    iterator = InfiniteIterator(loader)

    rdict = {"loss_values": [], "content_indices": []}

    for m in args.modalities:
        rdict[f"hz_{m}"] = []  # initialize for learned representations
        rdict[f"labels_{m}"] = {v: [] for v in args.DATASETCLASS.FACTORS[m].values()}
        rdict[f"hz_{m}_subsets"] = {s: [] for s in args.subsets}  # selected hz dimensions

    i = 0
    num_samples = num_samples or len(dataset)
    with torch.no_grad():
        while i < num_samples:
            # load batch
            i += loader.batch_size
            data = next(iterator)  # contains images, texts, and labels

            # compute loss
            loss_value, estimated_content_indices = val_step(data, fs, loss_func, args=args)

            rdict["loss_values"].append([loss_value])

            # collect representations
            for m_midx, m in enumerate(args.modalities):
                samples = data[m]  # Shape: [n_views, batch_size, ...]
                hz_m = fs[m_midx](torch.concat(samples, 0)).detach().cpu().numpy()
                rdict[f"hz_{m}"].append(hz_m)  # [n_views*batch_size, *text_dims]

                # collect image labels
                # data["z_image", "z_text"]: list of latent_dict, n_list = len(imgs)
                for k in rdict[f"labels_{m}"]:
                    labels_k = torch.concat([data[f"z_{m}"][i][k] for i in range(len(samples))], 0)
                    rdict[f"labels_{m}"][k].append(labels_k)

                for s_id, s in enumerate(args.subsets):
                    if len(args.subsets) == 1:  # there is only one content block to consider
                        rdict[f"hz_{m}_subsets"][s].append(hz_m)
                    else:
                        rdict[f"hz_{m}_subsets"][s].append(hz_m[..., estimated_content_indices[s_id]])

            del data

            rdict["content_indices"] += [estimated_content_indices]
    # concatenate each list of values along the batch dimension
    for k, v in rdict.items():
        if isinstance(v, list) and k != "content_indices":
            rdict[k] = np.concatenate(v, axis=0)
        elif isinstance(v, dict):
            for k2, v2 in v.items():
                rdict[k][k2] = np.concatenate(v2, axis=0)
    # rdict: hz_m_subsets: key: subset, values: selected "content" results
    return rdict


def main(args: argparse.Namespace):
    # create save_dir, where the model/results are or will be saved
    if args.dataset_name != "mpi3d":
        args.datapath = os.path.join(args.dataroot, args.dataset_name)
    else:
        # mpi3d does not have separate train;test;val
        args.datapath = os.path.join(args.dataroot, f"{args.dataset_name}/real3d_complicated_shapes_ordered.npz")
    # update model dir with dataset name
    args.model_dir = os.path.join(args.model_dir, args.dataset_name)
    if args.model_id is None:
        setattr(args, "model_id", uuid.uuid4())
    args.save_dir = os.path.join(args.model_dir, args.model_id)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # optionally, reuse existing arguments from settings.json (only for evaluation)
    if args.evaluate and args.load_args:
        with open(os.path.join(args.save_dir, "settings.json"), "r") as fp:
            loaded_args = json.load(fp)
        arguments_to_load = ["encoding_size", "hidden_size"]
        for arg in arguments_to_load:
            setattr(args, arg, loaded_args[arg])

    args = update_args(args)

    # print args
    print("Arguments:")
    for k, v in vars(args).items():
        print(f"\t{k}: {v}")

    # set all seeds
    if args.seed is not None:
        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    # save args to disk (only for training)
    if not args.evaluate:
        settings_dict = args.__dict__.copy()
        settings_dict.pop("DATASETCLASS")
        # writing to file
        with open(os.path.join(args.save_dir, "settings.json"), "w") as f:
            json.dump(settings_dict, f, indent=4)

    # set device
    if torch.cuda.is_available() and not args.no_cuda:
        device = f"cuda:{device_ids[0]}"
    else:
        device = "cpu"
        warnings.warn("cuda is not available or --no-cuda was set.")

    # define similarity metric and loss function
    sim_metric = torch.nn.CosineSimilarity(dim=-1)
    criterion = torch.nn.CrossEntropyLoss()

    def loss_func(z_rec_tuple, estimated_content_indices, subsets):
        return infonce_loss(
            z_rec_tuple,
            sim_metric=sim_metric,
            criterion=criterion,
            tau=args.tau,
            projector=(lambda x: x),
            # invertible_network_utils.construct_invertible_mlp(n=args.encoding_size, n_layers=2).to(device),
            estimated_content_indices=estimated_content_indices,
            subsets=subsets,
        )

    # define augmentations (only normalization of the input images)
    faiss.omp_set_num_threads(args.faiss_omp_threads)
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(args.DATASETCLASS.mean_per_channel, args.DATASETCLASS.std_per_channel),
        ]
    )

    # define kwargs
    dataset_kwargs = {"transform": transform}
    dataloader_kwargs = {
        "batch_size": args.batch_size,
        "shuffle": True,
        "drop_last": True,
        "num_workers": args.workers,
        "pin_memory": True,
    }
    train_dataset = args.DATASETCLASS(
        data_dir=args.datapath,
        mode="train",
        change_lists=args.change_lists,
        **dataset_kwargs,
    )
    if args.dataset_name == "multimodal3di":
        dataset_kwargs["vocab_filepath"] = train_dataset.vocab_filepath
    if args.dataset_name in ["mpi3d"]:
        dataset_kwargs["collate_random_pair"] = True
        train_dataset.collate_random_pair = True
        if args.collate_random_pair:
            dataloader_kwargs["collate_fn"] = train_dataset.__collate_fn__random_pair__

    # define datasets and dataloaders
    if args.evaluate:
        val_dataset = args.DATASETCLASS(
            data_dir=args.datapath,
            mode="val",
            change_lists=args.change_lists,
            **dataset_kwargs,
        )
        test_dataset = args.DATASETCLASS(
            data_dir=args.datapath,
            mode="test",
            change_lists=args.change_lists,
            **dataset_kwargs,
        )
    else:
        train_loader = DataLoader(train_dataset, **dataloader_kwargs)
        train_iterator = InfiniteIterator(train_loader)

    # define image encoder
    encoder_img = torch.nn.Sequential(
        resnet18(num_classes=args.hidden_size),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(args.hidden_size, args.encoding_size),
    )
    encoder_img = torch.nn.DataParallel(encoder_img, device_ids=device_ids)
    encoder_img.to(device)

    encoders = [encoder_img]

    if "text" in args.modalities:
        # define text encoder
        sequence_length = train_dataset.max_sequence_length
        encoder_txt = TextEncoder2D(
            input_size=train_dataset.vocab_size,
            output_size=args.encoding_size,
            sequence_length=sequence_length,
        )
        encoder_txt = torch.nn.DataParallel(encoder_txt, device_ids=device_ids)
        encoder_txt.to(device)
        encoders += [encoder_txt]

    # for evaluation, always load saved encoders
    if args.evaluate:
        for m_idx, m in enumerate(args.modalities):
            path = os.path.join(args.save_dir, f"encoder_{m}.pt")
            encoders[m_idx].load_state_dict(torch.load(path, map_location=device))

    # define the optimizer
    params = []
    for f in encoders:
        params += list(f.parameters())
    optimizer = torch.optim.Adam(params, lr=args.lr)

    # training
    # --------
    file_name = os.path.join(args.save_dir, "Training.csv")  # record the training loss
    if not args.evaluate:
        # training loop
        step = 1
        loss_values = []  # list to keep track of loss values
        while step <= args.train_steps:
            # training step
            data = next(train_iterator)  # contains images, texts, and labels
            loss_value, _ = train_step(data, encoders, loss_func, optimizer, params, args=args)
            loss_values.append(loss_value)

            # print average loss value
            if step % args.log_steps == 1 or step == args.train_steps:
                print(
                    f"Step: {step} \t",
                    f"Loss: {loss_value:.4f} \t",
                    f"<Loss>: {np.mean(loss_values[-args.log_steps:]):.4f} \t",
                )
                with open(f"{file_name}", "a+") as fileobj:
                    writer = csv.writer(fileobj)
                    wri = [
                        "Step",
                        f"{step}",
                        "<Loss>",
                        f"{np.mean(loss_values[-args.log_steps:]):.3f}",
                    ]
                    writer.writerow(wri)
                # fileobj.close()

            # save models and intermediate checkpoints
            if step % args.checkpoint_steps == 1 or step == args.train_steps or step == args.log_steps * 2:
                for m_idx, m in enumerate(args.modalities):
                    torch.save(
                        encoders[m_idx].state_dict(),
                        os.path.join(args.save_dir, f"encoder_{m}.pt"),
                    )

                if args.save_all_checkpoints:
                    torch.save(
                        encoders[m_idx].state_dict(),
                        os.path.join(args.save_dir, f"encoder_{m}_%d.pt" % step),
                    )
            step += 1

    # evaluation
    # ----------
    if args.evaluate:
        # collect encodings and labels from the validation and test data
        val_dict = get_data(
            val_dataset,
            encoders,
            loss_func,
            dataloader_kwargs,
            args=args,
            num_samples=args.val_size,
        )
        test_dict = get_data(
            test_dataset,
            encoders,
            loss_func,
            dataloader_kwargs,
            args=args,
            num_samples=args.test_size,
        )

        # print average loss values
        print(f"<Val Loss>: {np.mean(val_dict['loss_values']):.4f}")
        print(f"<Test Loss>: {np.mean(test_dict['loss_values']):.4f}")

        # handle edge case when the encodings are 1-dimensional
        if args.encoding_size == 1:
            val_dict[f"hz_{m}"] = val_dict[f"hz_{m}"].reshape(-1, 1)
            test_dict[f"hz_{m}"] = test_dict[f"hz_{m}"].reshape(-1, 1)

        # standardize the encodings
        for m in args.modalities:
            scaler = StandardScaler()
            val_dict[f"hz_{m}"] = scaler.fit_transform(val_dict[f"hz_{m}"])
            test_dict[f"hz_{m}"] = scaler.transform(test_dict[f"hz_{m}"])
            for s in args.subsets:
                scaler = StandardScaler()
                val_dict[f"hz_{m}_subsets"][s] = scaler.fit_transform(val_dict[f"hz_{m}_subsets"][s])
                test_dict[f"hz_{m}_subsets"][s] = scaler.transform(test_dict[f"hz_{m}_subsets"][s])

        # evaluate how well each factor can be predicted from the encodings
        results = []
        for m_idx, m in enumerate(args.modalities):
            factors_m = args.DATASETCLASS.FACTORS[m]
            discrete_factors_m = args.DATASETCLASS.DISCRETE_FACTORS[m]

            if args.eval_dci:
                # compute dci scores
                def repr_fn(samples):
                    f = encoders[m_idx]
                    # imgs: np array: [bs, 64, 64, 3]; text: ...
                    if m == "image" and args.dataset_name == "mpi3d":
                        samples = torch.stack([transform(i) for i in samples], dim=0)
                    with torch.no_grad():
                        hz = f(samples)
                    return hz.cpu().numpy()

                # compute DCI scores
                dci_score = dci.compute_dci(
                    ground_truth_data=val_dataset,
                    representation_function=repr_fn,
                    num_train=10000,
                    num_test=5000,
                    random_state=np.random.RandomState(),
                    factor_types=["discrete" if ix in discrete_factors_m else "continuous" for ix in factors_m],
                )
                # Open the CSV file with write permission
                with open(os.path.join(args.save_dir, f"dci_{m}.csv"), "w", newline="") as csvfile:
                    # Create a CSV writer using the field/column names
                    writer = csv.DictWriter(csvfile, fieldnames=dci_score.keys())
                    # Write the header row (column names)
                    writer.writeheader()
                    # Write the data
                    writer.writerow(dci_score)
                continue

            for ix, factor_name in factors_m.items():
                for s in args.subsets:
                    # select data
                    train_inputs = val_dict[f"hz_{m}_subsets"][s]
                    test_inputs = test_dict[f"hz_{m}_subsets"][s]
                    train_labels = val_dict[f"labels_{m}"][factor_name]
                    test_labels = test_dict[f"labels_{m}"][factor_name]
                    data = [train_inputs, train_labels, test_inputs, test_labels]

                    # append results
                    results.append(eval_step(ix, s, m, factor_name, discrete_factors_m, data))
                # independent component extraction
                if args.eval_style and len(args.style_indices) > 0:
                    # select data
                    train_inputs = val_dict[f"hz_{m}"][..., args.style_indices]
                    test_inputs = test_dict[f"hz_{m}"][..., args.style_indices]
                    train_labels = val_dict[f"labels_{m}"][factor_name]
                    test_labels = test_dict[f"labels_{m}"][factor_name]
                    data = [train_inputs, train_labels, test_inputs, test_labels]
                    # append results
                    results.append(eval_step(ix, (-1), m, factor_name, discrete_factors_m, data))

        # convert evaluation results into tabular form
        columns = [
            "subset",
            "ix",
            "modality",
            "factor_name",
            "factor_type",
            "r2_linreg",
            "r2_krreg",
            "acc_logreg",
            "acc_mlp",
        ]
        df_results = pd.DataFrame(results, columns=columns)
        df_results.to_csv(os.path.join(args.save_dir, "results.csv"))
        print(df_results.to_string())


def eval_step(ix, subset, modality, factor_name, discrete_factors_m, data):
    """
    Evaluate the performance of a factor prediction model for a given factor.

    Args:
        ix (int): The index of the factor.
        subset (str): The subset name.
        modality (str): The modality name.
        factor_name (str): The name of the factor.
        discrete_factors_m (list): A list of indices of discrete factors for the modality.
        data (tuple): A tuple containing the input features and target values.

    Returns:
        list: A list containing the evaluation results, including R2 scores and accuracy.

    """
    r2_linreg, r2_krreg, acc_logreg, acc_mlp = [np.nan] * 4

    # check if factor ix is discrete for modality m
    if ix in discrete_factors_m:
        factor_type = "discrete"
    else:
        factor_type = "continuous"

    # for continuous factors, do regression and compute R2 score
    if factor_type == "continuous":
        # linear regression
        linreg = LinearRegression(n_jobs=-1)
        r2_linreg = utils.evaluate_prediction(linreg, r2_score, *data)
        if args.grid_search_eval:
            # nonlinear regression # usually a bit compute-heavy here
            gskrreg = GridSearchCV(
                KernelRidge(kernel="rbf", gamma=0.1),
                param_grid={
                    "alpha": [1e0, 0.1, 1e-2, 1e-3],
                    "gamma": np.logspace(-2, 2, 4),
                },
                cv=3,
                n_jobs=-1,
            )
            r2_krreg = utils.evaluate_prediction(gskrreg, r2_score, *data)
        # NOTE: MLP is a lightweight alternative
        r2_krreg = utils.evaluate_prediction(MLPRegressor(max_iter=1000), r2_score, *data)

    # for discrete factors, do classification and compute accuracy
    if factor_type == "discrete" and factor_name != "object_zpos":
        # we disable prediction on zpos in m3di because it is constant
        # logistic classification
        logreg = LogisticRegression(n_jobs=-1, max_iter=1000)
        acc_logreg = utils.evaluate_prediction(logreg, accuracy_score, *data)
        # nonlinear classification
        mlpreg = MLPClassifier(max_iter=1000)
        acc_mlp = utils.evaluate_prediction(mlpreg, accuracy_score, *data)

    res_row = [
        subset,
        ix,
        modality,
        factor_name,
        factor_type,
        r2_linreg,
        r2_krreg,
        acc_logreg,
        acc_mlp,
    ]
    return res_row


if __name__ == "__main__":
    # parse args
    #         argparser object
    #            |          do arg parsing
    #            V             v
    args = parse_args().parse_args()
    main(args)
