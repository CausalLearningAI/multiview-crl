"""
Collection of datasets.
"""

import io
import json
import os
from abc import abstractmethod
from collections import Counter, OrderedDict
from typing import Callable, Optional

import faiss
import numpy as np
import pandas as pd
import torch
from nltk.tokenize import sent_tokenize, word_tokenize
from torchvision.datasets.folder import pil_loader

import spaces
from spaces import NBoxSpace


class OrderedCounter(Counter, OrderedDict):
    """Counter that remembers the order of elements encountered."""

    def __repr__(self):
        return "%s(%r)" % (self.__class__.__name__, OrderedDict(self))

    def __reduce__(self):
        return self.__class__, (OrderedDict(self),)


class MultiviewDataset(torch.utils.data.Dataset):
    """
    A dataset class for handling multiview data.

    Attributes:
        FACTORS (dict): A dictionary mapping factor names to their corresponding indices.
        DISCRETE_FACTORS (list): A list of discrete factor names.
        FACTOR_SIZES (dict): A dictionary mapping factor names to their sizes.
        LATENT_SPACES (dict): A dictionary mapping factor names to their corresponding latent spaces.
        mean_per_channel (list): A list of mean values per channel.
        std_per_channel (list): A list of standard deviation values per channel.
    """

    FACTORS = None
    DISCRETE_FACTORS = None
    FACTOR_SIZES = None
    LATENT_SPACES = None

    mean_per_channel = [0.0] * 3
    std_per_channel = [1.0] * 3

    @staticmethod
    def __construct_index__(latents, approximate_mode=True):
        """
        Construct an index for approximate nearest neighbor search.

        Args:
            latents (numpy.ndarray): The latent vectors.
            approximate_mode (bool): Whether to use approximate mode or not.

        Returns:
            faiss.Index: The constructed index.
        """
        if approximate_mode:
            _index = faiss.index_factory(latents.shape[1], "IVF1024_HNSW32,Flat")
            _index.efSearch = 8
            _index.nprobe = 10
        else:
            _index = faiss.IndexFlatL2(latents.shape[1])

        if approximate_mode:
            _index.train(latents)
        _index.add(latents)
        return _index

    @staticmethod
    def __search_view__(z, _index, latents, image_paths, transform, loader, augment=False, idx_original=None):
        """
        Search for a view in the dataset.

        Args:
            z (torch.Tensor): The query latent vector.
            _index (faiss.Index): The index for nearest neighbor search.
            latents (numpy.ndarray): The latent vectors.
            image_paths (list): The paths to the images.
            transform (torchvision.transforms.Compose): The image transformation pipeline.
            loader (torchvision.datasets.folder.default_loader): The image loader function.
            augment (bool): Whether to search for an augmented view or not.
            idx_original (int): The index of the original view.

        Returns:
            tuple: A tuple containing the index, latent vector, and image of the found view.
        """
        if not augment:
            distance_z, index_z = _index.search(z.numpy(), 1)  # looking for the original view
            index_z = index_z[0, 0]
        else:  # looking for augmented view
            if z.ndim > 1:
                z = z.squeeze()
            # assert isinstance(idx_original, int), 'original index must be given for augmented view'
            distance_z, index_z = _index.search(z[None].numpy() if isinstance(z, torch.Tensor) else z[None], 2)
            # don't use the same sample for z, z~
            if index_z[0, 0] != idx_original:
                index_z = index_z[0, 0]
            else:
                index_z = index_z[0, 1]
        z = latents[index_z]
        path_z = image_paths[index_z]
        img = transform(loader(path_z))

        return index_z, z, img

    @abstractmethod
    def __getview__(self, item):
        """
        Get a view from the dataset.

        Args:
            item: The index of the view.

        Returns:
            tuple: A tuple containing the index, latent vector, and image of the view.
        """
        raise NotImplementedError

    @abstractmethod
    def __get_augmented_view__(self, idx, z, change_list):
        """
        Get an augmented view from the dataset.

        Args:
            idx: The index of the view.
            z (dict): The latent vectors.
            change_list (list): The list of factors to be changed.

        Returns:
            tuple: A tuple containing the index, latent vector, and image of the augmented view.
        """
        raise NotImplementedError

    @abstractmethod
    def sample(self, size, random_state=None):
        """
        Sample views from the dataset.

        Args:
            size (int): The number of views to sample.
            random_state (int): The random seed for reproducibility.

        Returns:
            list: A list of sampled views.
        """
        raise NotImplementedError

    @classmethod
    def sample_style_indices(cls, num_factors):
        """
        Sample style indices for perturbing latent factors.

        Args:
            num_factors (int): The number of factors.

        Returns:
            list: A list of style indices.
        """
        # perturb a random number of latents
        style_size = np.random.randint(1, num_factors)
        index_list = np.random.choice(num_factors, np.random.choice([1, style_size]), replace=False)
        if cls == "Causal3DIdent":
            return index_list + 1  # this is because the latent dim does not include class;
            # images are stored in different class folders thus the class information is given.
        else:
            return index_list

    def __collate_fn__random_pair__(self, batch):
        """
        Collate function for creating random pairs of views.

        Args:
            batch (list): A list of views.

        Returns:
            dict: A dictionary containing the collated batch.
        """
        # Following Locatello2020weakly; with 50% chance only perturb one latent;
        # with another 50% chance perturb a random number of latents.
        imgs, aug_imgs = [], []
        factors = self.FACTORS["image"]

        z_images = {k: [] for k in factors.values()}  # key are the name of the latent factors, not the idx
        aug_z_imgs = {k: [] for k in factors.values()}

        num_factors = len(factors)

        index_list = self.sample_style_indices(num_factors)
        factor_keys = np.asarray(list(factors.keys()))
        content_indices = [factor for factor in factor_keys if factor not in index_list]
        for b in batch:
            img = b["image"]  # [3, 64, 64]
            idx = b["index"]  # [7, ]
            z_image = b["z_image"]  # dict: 7 keys, 7 values

            # compute augmented view
            aug_idx, aug_z_img, aug_img = self.__get_augmented_view__(idx, z_image, change_list=index_list)

            # append sample and augemented data to batch
            imgs += [img]
            aug_imgs += [aug_img]
            for idx, k in factors.items():
                z_images[k] += [z_image[k]]
                aug_z_imgs[k] += [aug_z_img[idx]]

        for k in factors.values():
            z_images[k] = torch.tensor(z_images[k])
            aug_z_imgs[k] = torch.tensor(aug_z_imgs[k])
        return {
            "image": [
                torch.stack(imgs, 0),
                torch.stack(aug_imgs, 0),
            ],
            "z_image": [z_images, aug_z_imgs],
            "content_indices": [content_indices],
        }


class MPI3D(MultiviewDataset):
    """
    MPI3D dataset class for multiview data.

    Attributes:
        FACTORS (dict): Dictionary mapping latent factors to their corresponding names.
        DISCRETE_FACTORS (dict): Dictionary mapping latent factors to their corresponding names,
                                 with all latents discretized.
        FACTOR_SIZES (list): List of sizes for each latent factor.
        LATENT_SPACES (dict): Dictionary mapping latent factors to their corresponding spaces.
        mean_per_channel (list): List of mean values per channel.
        std_per_channel (list): List of standard deviation values per channel.

    Methods:
        __init__: Initializes the MPI3D dataset.
        __len__: Returns the number of samples in the dataset.
        __getview__: Returns the view at the specified index.
        __get_augmented_view__: Returns the augmented view at the specified index.
        __getitem__: Returns the item at the specified index.
        sample: Samples latent indices and corresponding images from the dataset.
    """

    FACTORS = {
        "image": {
            0: "object_color",
            1: "object_shape",
            2: "object_size",
            3: "camera_height",
            4: "background_color",
            5: "horizontal_axis",
            6: "vertical_axis",
        }
    }
    DISCRETE_FACTORS = FACTORS.copy()  # all latents are discretized

    # define latent spaces for each component
    FACTOR_SIZES = [4, 4, 2, 3, 3, 40, 40]  # first 7 latent factors, last 3 img # (460,800, 64, 64, 3)
    LATENT_SPACES = {"image": {}}
    for i, v in FACTORS["image"].items():
        LATENT_SPACES["image"][i] = spaces.DiscreteSpace(FACTOR_SIZES[i])

    mean_per_channel = [0.0993, 0.1370, 0.1107]  # values from MPI3d-realworld complex
    std_per_channel = [0.0945, 0.0935, 0.0887]  # values from MPI3_real-world complex

    def __init__(
        self,
        data_dir,
        n_view: int = 1,
        mode="train",
        transform: Optional[Callable] = None,
        collate_random_pair=True,
        change_lists=None,
    ):
        """
        Initializes the MPI3D dataset.

        Args:
            data_dir (str): The directory path to the dataset.
            n_view (int): The number of views.
            mode (str): The mode of the dataset (e.g., "train", "test").
            transform (Optional[Callable]): Optional transform function to be applied to the data.
            collate_random_pair (bool): Whether to collate random pairs of images.

        Returns:
            None.
        """
        npz = np.load(data_dir, allow_pickle=True)
        data = npz["images"]  # NOTE: you cannot shuffle the data otherwise the dim<->latent will be messed up!
        self.num_samples = len(data)

        self.data_dir = data_dir
        self.n_view = n_view
        self.mode = mode
        self.collate_random_pair = collate_random_pair

        setattr(MPI3D, "data", data.reshape(self.FACTOR_SIZES + [64, 64, 3]))
        setattr(MPI3D, "transform", transform or (lambda x: x))

    def __len__(self):
        """
        Returns the number of samples in the dataset.

        Returns:
            int: The number of samples in the dataset.
        """
        return self.num_samples

    def __getview__(self, item):
        """
        Returns the view at the specified index.

        Args:
            item (int): The index of the view.

        Returns:
            tuple: A tuple containing the index, z, and image of the view.
        """
        index = np.unravel_index(item, shape=self.FACTOR_SIZES)  # multi-dimensional index
        return index, index, self.transform(self.data[index])  # index, z, img

    def __get_augmented_view__(self, idx, z, change_list):
        """
        Returns the augmented view at the specified index.

        Args:
            idx (tuple): The index of the view.
            z (tuple): The z value of the view.
            change_list (list): The list of indices to change.

        Returns:
            tuple: A tuple containing the augmented index, z, and image of the view.
        """
        aug_idx = tuple(
            int(space.uniform(original=idx[i], size=1).item()) if i in change_list else idx[i]
            for i, space in MPI3D.LATENT_SPACES["image"].items()
        )
        return aug_idx, aug_idx, self.transform(self.data[*aug_idx])  # idx, z, x

    def __getitem__(self, idx):
        """
        Returns the item at the specified index.

        Args:
            idx (int): The index of the item.

        Returns:
            dict: A dictionary containing the image, index, and z_image.
        """
        index, z_image, image = self.__getview__(idx)
        if self.collate_random_pair:
            return {
                "image": image,
                "index": index,
                "z_image": {v: z_image[k] for k, v in self.FACTORS["image"].items()},
            }
        else:
            raise NotImplementedError  # for this you need to pre-define a change list

    def sample(self, size, random_state=None):
        """
        Samples latent indices and corresponding images from the dataset.

        Args:
            size (int): The number of samples to be generated.
            random_state (Optional): The random state to be used for sampling.

        Returns:
            tuple: A tuple containing the latent indices and corresponding images.
        """
        # Shape: [num_factors, size]
        latent_idx = np.vstack(
            [space.uniform(size=size).int().numpy() for i, space in MPI3D.LATENT_SPACES["image"].items()]
        )

        return latent_idx.T, np.stack([self.data[tuple(latent_idx[:, i])] for i in range(size)])


# --------------------------------- Independent MPI3D -------------------------------
class Indepdenent3DIdent(MultiviewDataset):
    """
    Dataset class for independent 3D object identification.

    Args:
        data_dir (str): The directory path where the data is stored.
        change_lists: The list of change lists.
        mode (str, optional): The mode of the dataset. Defaults to "train".
        transform (Callable, optional): The transformation function to apply to the data. Defaults to None.
        loader (Callable, optional): The image loader function. Defaults to pil_loader.
        approximate_mode (bool, optional): Whether to use approximate mode. Defaults to True.
        latent_dimensions_to_use (list, optional): The list of latent dimensions to use. Defaults to None.
        collate_random_pair (bool, optional): Whether to collate random pairs. Defaults to False.
    """

    FACTORS = {
        "image": {
            0: "object_xpos",
            1: "object_ypos",
            2: "object_zpos",
            3: "object_alpharot",
            4: "object_betarot",
            5: "object_gammarot",
            6: "object_color",
            7: "background_color",
            8: "spotlight_pos",
            9: "spotlight_color",
        }
    }
    DISCRETE_FACTORS = {"image": {}}  # no discrete factors

    # construct latent spaces, all continous on a hyper-cube
    LATENT_SPACES = {"image": {}}
    for i, v in FACTORS["image"].items():
        LATENT_SPACES["image"][i] = NBoxSpace(n=1, min_=-1.0, max_=1.0)

    POSITIONS = [0, 1, 2, 8]
    ROTATIONS = [3, 4, 5]
    HUES = [6, 7, 9]

    mean_per_channel = [0.4363, 0.2818, 0.3045]
    std_per_channel = [0.1197, 0.0734, 0.0919]

    def __init__(
        self,
        data_dir: str,
        change_lists,
        mode="train",
        transform: Optional[Callable] = None,
        loader: Optional[Callable] = pil_loader,
        approximate_mode: Optional[bool] = True,
        latent_dimensions_to_use=None,
        collate_random_pair=False,
    ) -> None:
        super(Indepdenent3DIdent, self).__init__()

        self.collate_random_pair = collate_random_pair

        self.mode = mode
        if self.mode == "val":
            self.mode = "test"

        root = os.path.join(data_dir, f"{self.mode}")
        self.latents = np.load(os.path.join(root, "raw_latents.npy"))
        self.unfiltered_latents = self.latents
        self.change_lists = change_lists

        if latent_dimensions_to_use is not None:
            self.latents = np.ascontiguousarray(self.latents[:, latent_dimensions_to_use])

        # self.sigma = 1.0 # for conditioned sampling
        self.transform = transform or (lambda x: x)

        max_length = int(np.ceil(np.log10(len(self.latents))))
        self.image_paths = [
            os.path.join(root, "images", f"{str(i).zfill(max_length)}.png") for i in range(self.latents.shape[0])
        ]
        self.loader = loader
        self._index = MultiviewDataset.__construct_index__(self.latents, approximate_mode=approximate_mode)

    def __len__(self) -> int:
        return len(self.latents)

    def __repr__(self) -> str:
        head = "Dataset " + self.__class__.__name__
        body = ["Number of datapoints: {}".format(len(self.latents))]
        if self.root is not None:
            body.append("Root location: {}".format(self.root))
        body += self.extra_repr().splitlines()
        if hasattr(self, "transforms") and self.transforms is not None:
            body += [repr(self.transforms)]
        lines = [head] + [" " * self._repr_indent + line for line in body]
        return "\n".join(lines)

    def __getview__(self, item):
        """
        Randomly sample a view.

        Args:
            item: The item index.

        Returns:
            The index, latent vector, and image of the sampled view.
        """
        del item
        z = np.stack(
            [space.uniform(size=1).squeeze() for space in Indepdenent3DIdent.LATENT_SPACES["image"].values()]
        ).flatten()
        return MultiviewDataset.__search_view__(
            z=torch.from_numpy(z[None]),
            _index=self._index,
            latents=self.latents,
            image_paths=self.image_paths,
            transform=self.transform,
            loader=self.loader,
            augment=False,
        )

    def __get_augmented_view__(self, idx, z, change_list):
        """
        Returns an augmented view of the dataset based on the given index and latent vector.

        Args:
            idx (int): The index of the view in the dataset.
            z (numpy.ndarray): The latent vector of the view.
            change_list (list): A list of indices indicating which elements of the latent vector should be changed.

        Returns:
            view: An augmented view with perturbed latents.
        """
        z_tilde = np.copy(z)
        for j in change_list:
            z_tilde[j] = (
                self.LATENT_SPACES["image"][j]
                .uniform(
                    size=1,
                    device="cpu",
                )
                .flatten()
            )
        return MultiviewDataset.__search_view__(
            z=z_tilde[None],
            _index=self._index,
            latents=self.latents,
            image_paths=self.image_paths,
            transform=self.transform,
            loader=self.loader,
            augment=True,
            idx_original=idx,
        )

    def __getitem__(self, item):
        """
        Retrieve an item from the dataset.

        Args:
            item (int): Index of the item to retrieve.

        Returns:
            dict: A dictionary containing the retrieved item, including the index, image, and z_image.

        Raises:
            IndexError: If the given item index is out of range.
        """
        # at first sample z
        # then map them to the closest grid point for which we have images
        index_z, z, x = self.__getview__(item)
        imgs = [x]
        zs = [z]
        indices = [index_z]

        if self.collate_random_pair:
            # when collating random pair, then the perturbed components vary for each batch,
            # the change list will be given in the collate_fn
            return {
                "index": index_z,
                "image": imgs,
                "z_image": [{self.FACTORS["image"][i]: v for i, v in enumerate(z)} for z in zs],
            }
        else:
            for k in range(len(self.change_lists)):
                index_z_tilde, z_tilde, x_tilde = self.__get_augmented_view__(
                    index_z, z, change_list=self.change_lists[k]
                )
                indices += [index_z_tilde]
                zs += [z_tilde]
                imgs += [x_tilde]
            return {
                "index": indices,
                "image": imgs,
                "z_image": [{self.FACTORS["image"][i]: v for i, v in enumerate(z)} for z in zs],
            }

    def sample(self, size, random_state=None):
        """
        Samples a batch of data from the dataset.

        Args:
            size (int): The number of samples to be generated.
            random_state (int, optional): Random seed for reproducibility. Defaults to None.

        Returns:
            Tuple[np.ndarray, torch.Tensor]: A tuple containing the sampled latents as a numpy array of shape
                                            (n_latents, size)
            and the corresponding images as a torch tensor of shape (size, channels, height, width).
        """

        latents = np.vstack(
            [space.uniform(size=size) for i, space in Indepdenent3DIdent.LATENT_SPACES.items()]
        )  # (size, n_latents)

        imgs = []
        for z in latents:
            _, _, img = MultiviewDataset.__search_view__(
                z,
                self._index,
                self.latents,
                self.image_paths,
                transform=self.transform,
                loader=self.loader,
                augment=False,
            )
            imgs += [img]
        return latents.T, torch.stack(imgs)


# ----------------------------------- Causal 3d ident --------------------------------
class Causal3DIdent(MultiviewDataset):
    """
    Dataset class for Causal3DIdent.

    Args:
        change_lists (List[List[int]]): List of change lists, where each change list is a list of indices representing
            the factors to be changed.
        data_dir (str): Directory path to the data.
        mode (str, optional): Mode of the dataset. Defaults to "train".
        transform (Callable, optional): Transform function to be applied to the images. Defaults to None.
        loader (Callable, optional): Loader function to load the images. Defaults to pil_loader.
        latent_dimensions_to_use (range, optional): Range of latent dimensions to use. Defaults to range(10).
        approximate_mode (bool, optional): Flag indicating whether to use approximate mode. Defaults to True.
    """

    FACTORS = {
        "image": {
            0: "object_shape",
            1: "object_ypos",
            2: "object_xpos",
            3: "object_zpos",
            4: "object_alpharot",
            5: "object_betarot",
            6: "object_gammarot",
            7: "spotlight_pos",
            8: "object_color",
            9: "spotlight_color",
            10: "background_color",
        }
    }
    CLASSES = range(7)  # number of object shapes
    LATENT_SPACES = {"image": {}}
    for i, v in FACTORS["image"].items():
        if i == 0:
            LATENT_SPACES["image"][i] = spaces.DiscreteSpace(n_choices=len(CLASSES))
        else:
            LATENT_SPACES["image"][i] = spaces.NBoxSpace(n=1, min_=-1.0, max_=1.0)

    mean_per_channel = [0.4327, 0.2689, 0.2839]
    std_per_channel = [0.1201, 0.1457, 0.1082]

    POSITIONS = [1, 2, 3]
    ROTATIONS = [4, 5, 6]
    HUES = [7, 8, 9]

    DISCRETE_FACTORS = {"image": {0: "object_shape"}}

    def __init__(
        self,
        change_lists,
        data_dir: str,
        mode: str = "train",
        transform: Optional[Callable] = None,
        loader: Optional[Callable] = pil_loader,
        latent_dimensions_to_use=range(10),
        approximate_mode: Optional[bool] = True,
    ):
        super(Causal3DIdent, self).__init__()

        self.change_lists = change_lists

        self.mode = mode
        if self.mode == "val":
            self.mode = "test"

        self.sigma = 1.0
        self.root = os.path.join(data_dir, self.mode)

        self.classes = self.CLASSES
        self.latent_classes = []
        for i in self.classes:
            self.latent_classes.append(np.load(os.path.join(self.root, "raw_latents_{}.npy".format(i))))
        self.unfiltered_latent_classes = self.latent_classes

        if latent_dimensions_to_use is not None:
            # print('not none')
            for i in self.classes:
                self.latent_classes[i] = np.ascontiguousarray(self.latent_classes[i][:, latent_dimensions_to_use])

        self.image_paths_classes = []
        for i in self.classes:
            max_length = int(np.ceil(np.log10(len(self.latent_classes[i]))))
            self.image_paths_classes.append(
                [
                    os.path.join(self.root, "images_{}".format(i), f"{str(j).zfill(max_length)}.png")
                    for j in range(self.latent_classes[i].shape[0])
                ]
            )
        self.loader = loader
        self.transform = transform or (lambda x: x)

        self._index_classes = []
        for i in self.classes:
            _index = MultiviewDataset.__construct_index__(
                latents=self.latent_classes[i], approximate_mode=approximate_mode
            )
            self._index_classes.append(_index)

    def __len__(self) -> int:
        return len(self.latent_classes[0]) * len(self.classes)

    def __getview__(self, item):
        """
        Randomly sample a view.

        Args:
            item (int): Index of the view.

        Returns:
            tuple: A tuple containing the item index, latent vector, latent dictionary, and transformed image.
        """
        class_id = item // len(self.latent_classes[0])
        in_class_id = item % len(self.latent_classes[0])
        z = self.latent_classes[class_id][in_class_id]
        # z.shape=(10, ), contains factors except object shape
        path_z = self.image_paths_classes[class_id][in_class_id]
        sample = self.loader(path_z)
        x1 = self.transform(sample)

        z_dict = {self.FACTORS["image"][0]: class_id}  # dictionary of latents including the object shape
        for i in range(len(z.flatten())):
            z_dict[self.FACTORS["image"][i + 1]] = z.flatten()[i]  # index 0 is used to store object shape
        return item, z, z_dict, x1

    def __get_augmented_view__(self, idx, z, change_list):
        """
        Get an augmented view.

        Args:
            idx (int): Index of the view.
            z (np.ndarray): Latent vector.
            change_list (List[int]): List of indices representing the factors to be changed.

        Returns:
            tuple: A tuple containing the index of the augmented view, latent dictionary, and transformed image.
        """
        class_id = idx // len(self.latent_classes[0])
        z_tilde = np.copy(z)
        for j in change_list:
            assert j > 0
            # only search within the same class, thus the first component (class) is not touched.
            z_tilde[j - 1] = self.LATENT_SPACES["image"][j].uniform(size=1, device="cpu").numpy().flatten()

        index_z_tilde, z_tilde, x_tilde = MultiviewDataset.__search_view__(
            z=z_tilde,
            _index=self._index_classes[class_id],
            latents=self.latent_classes[class_id],
            image_paths=self.image_paths_classes[class_id],
            transform=self.transform,
            loader=self.loader,
            augment=True,
            idx_original=idx % len(self.latent_classes[0]),  # input the in-class index;
        )

        # reformat the latents into dictionary
        z_tilde_dict = {self.FACTORS["image"][0]: class_id}
        for i in range(1, len(z_tilde.flatten())):
            # optional manually fix invariance here
            z_tilde_dict[self.FACTORS["image"][i]] = z_tilde.flatten()[i] if i in change_list else z.flatten()[i]

        return index_z_tilde, z_tilde_dict, x_tilde

    def __getitem__(self, item):
        """
        Retrieves the item at the given index from the dataset.

        Parameters:
            item (int): The index of the item to retrieve.

        Returns:
            dict: A dictionary containing the retrieved images and z_images.
                - "image" (list): A list of images.
                - "z_image" (list): A list of z_images.
        """

        _, z, z_dict, x = self.__getview__(item)
        z_dicts = [z_dict]
        images = [x]
        for k in range(len(self.change_lists)):
            change_list = self.change_lists[k]
            _, z_tilde_dict, x_tilde = self.__get_augmented_view__(idx=item, z=z, change_list=change_list)
            z_dicts += [z_tilde_dict]
            images += [x_tilde]
        return {"image": images, "z_image": z_dicts}


# ----------------------------------- Multimodal3DIdent --------------------------------
class Multimodal3DIdent(MultiviewDataset):
    """Multimodal3DIdent Dataset.

    Attributes:
        FACTORS (dict): names of factors for image and text modalities.
        DISCRETE_FACTORS (dict): names of discrete factors, respectively.
    """

    FACTORS = {
        "image": {
            0: "object_shape",
            1: "object_ypos",
            2: "object_xpos",
            3: "object_zpos",  # is constant
            4: "object_alpharot",
            5: "object_betarot",
            6: "object_gammarot",
            7: "spotlight_pos",
            8: "object_color",
            9: "spotlight_color",
            10: "background_color",
        },
        "text": {
            0: "object_shape",
            1: "object_ypos",
            2: "object_xpos",
            3: "object_zpos",  # is constant
            4: "object_color_index",
            5: "text_phrasing",
        },
    }

    DISCRETE_FACTORS = {
        "image": {
            0: "object_shape",
            1: "object_ypos",
            2: "object_xpos",
            3: "object_zpos",  # is constant
        },
        "text": {
            0: "object_shape",
            1: "object_ypos",
            2: "object_xpos",
            3: "object_zpos",  # is constant
            4: "object_color_index",
            5: "text_phrasing",
        },
    }
    LATENT_SPACES = {}
    LATENT_SPACES["image"] = {
        0: spaces.DiscreteSpace(n_choices=7),
        1: spaces.DiscreteSpace(n_choices=3),
        2: spaces.DiscreteSpace(n_choices=3),
        3: spaces.DiscreteSpace(n_choices=1),
        4: spaces.NBoxSpace(n=1, min_=0.0, max_=1.0),
        5: spaces.NBoxSpace(n=1, min_=0.0, max_=1.0),
        6: spaces.NBoxSpace(n=1, min_=0.0, max_=1.0),
        7: spaces.NBoxSpace(n=1, min_=0.0, max_=1.0),
        8: spaces.NBoxSpace(n=1, min_=0.0, max_=1.0),
        9: spaces.NBoxSpace(n=1, min_=0.0, max_=1.0),
        10: spaces.NBoxSpace(n=1, min_=0.0, max_=1.0),
    }
    mean_per_channel = [0.4327, 0.2689, 0.2839]  # values from Causal3DIdent
    std_per_channel = [0.1201, 0.1457, 0.1082]  # values from Causal3DIdent

    def __init__(
        self,
        data_dir,
        change_lists,
        mode="train",
        has_labels=True,
        vocab_filepath=None,
        transform: Optional[Callable] = None,
        loader: Optional[Callable] = pil_loader,
        approximate_mode: Optional[bool] = True,
    ):
        """
        Args:
            data_dir (string): path to  directory.
            mode (string): name of data split, 'train', 'val', or 'test'.
            transform (callable): Optional transform to be applied.
            has_labels (bool): Indicates if the data has ground-truth labels.
            vocab_filepath (str): Optional path to a saved vocabulary. If None,
              the vocabulary will be (re-)created.
        """
        assert has_labels, "must have latent labels"
        self.mode = mode
        self.transform = transform
        self.has_labels = has_labels
        self.data_dir = data_dir
        self.data_dir_mode = os.path.join(data_dir, mode)
        self.latents_text_filepath = os.path.join(self.data_dir_mode, "latents_text.csv")
        self.latents_image_filepath = os.path.join(self.data_dir_mode, "latents_image.csv")
        self.text_filepath = os.path.join(self.data_dir_mode, "text", "text_raw.txt")
        self.image_dir = os.path.join(self.data_dir_mode, "images")

        # load text
        text_in_sentences, text_in_words = self._load_text()
        self.text_in_sentences = text_in_sentences  # sentence-tokenized text
        self.text_in_words = text_in_words  # word-tokenized text

        # determine num_samples and max_sequence_lengt
        self.num_samples = len(self.text_in_sentences)
        self.max_sequence_length = max([len(sent) for sent in self.text_in_words]) + 1  # +1 for "eos"

        # load or create the vocabulary (i.e., word <-> index maps)
        self.w2i, self.i2w = self._load_vocab(vocab_filepath)
        self.vocab_size = len(self.w2i)
        if vocab_filepath:
            self.vocab_filepath = vocab_filepath
        else:
            self.vocab_filepath = os.path.join(self.data_dir, "vocab.json")

        # optionally, load ground-truth labels
        if has_labels:
            self.labels = self._load_labels()
            self.z_image = self.labels["z_image"]

        # create list of image filepaths
        image_paths = []
        width = int(np.ceil(np.log10(self.num_samples)))
        for i in range(self.num_samples):
            fp = os.path.join(self.image_dir, str(i).zfill(width) + ".png")
            image_paths.append(fp)
        self.image_paths = image_paths

        # perturbed latent variables
        self.change_lists = change_lists
        self._index = MultiviewDataset.__construct_index__(
            latents=self.z_image.values, approximate_mode=approximate_mode
        )
        self.loader = loader
        self.transform = transform or (lambda x: x)

    def get_w2i(self, word):
        """
        Retrieves the index of a given word from the word-to-index dictionary.

        Args:
            word (str): The word to retrieve the index for.

        Returns:
            int or str: The index of the word if it exists in the dictionary, or "{unk}" if the word is unknown.
        """
        try:
            return self.w2i[word]
        except KeyError:
            return "{unk}"  # special token for unknown words

    def _load_text(self):
        """
        Load and preprocess text data.

        Returns:
            text_in_sentences (list): A list of sentences in the text.
            text_in_words (list): A list of words in the text.
        """

        print(f"Tokenization of {self.mode} data...")

        # load raw text
        with open(self.text_filepath, "r") as f:
            text_raw = f.read()

        # create sentence-tokenized text
        text_in_sentences = sent_tokenize(text_raw)

        # create word-tokenized text
        text_in_words = [word_tokenize(sent) for sent in text_in_sentences]

        return text_in_sentences, text_in_words

    def _load_labels(self):
        """
        Load image and text labels from CSV files and create a label dictionary.

        Returns:
            dict: A dictionary containing the loaded image and text labels.
                The dictionary has two keys: "z_image" and "z_text".
                The values associated with these keys are pandas DataFrames
                containing the loaded labels.
        """
        # load image labels
        z_image = pd.read_csv(self.latents_image_filepath)

        # load text labels
        z_text = pd.read_csv(self.latents_text_filepath)

        # check if all factors are present
        for v in self.FACTORS["image"].values():
            assert v in z_image.keys()
        for v in self.FACTORS["text"].values():
            assert v in z_text.keys()

        # create label dict
        labels = {"z_image": z_image, "z_text": z_text}

        return labels

    def _load_labels_by_class(self):
        """
        Load image labels and group them by class.

        Returns:
            z_image_by_class (pandas.core.groupby.DataFrameGroupBy): Grouped image labels by class.
        """
        # load image labels
        z_image = pd.read_csv(self.latents_image_filepath)
        # groupby object shape (or we call it class)
        z_image_by_class = z_image.groupby("object_shape")
        return z_image_by_class

    def _create_vocab(self, vocab_filepath):
        """
        Create a vocabulary from the training data and save it to a file.

        Args:
            vocab_filepath (str): The file path to save the vocabulary.

        Returns:
            dict: The vocabulary dictionary containing word-to-index and index-to-word mappings.
        """

        print(f"Creating vocabulary as '{vocab_filepath}'...")

        if self.mode != "train":
            raise ValueError("Vocabulary should be created from training data")

        # initialize counter and word <-> index maps
        ordered_counter = OrderedCounter()  # counts occurrence of each word
        w2i = dict()  # word-to-index map
        i2w = dict()  # index-to-word map
        unique_words = []

        # add special tokens for padding, end-of-string, and unknown words
        special_tokens = ["{pad}", "{eos}", "{unk}"]
        for st in special_tokens:
            i2w[len(w2i)] = st
            w2i[st] = len(w2i)

        for i, words in enumerate(self.text_in_words):
            ordered_counter.update(words)

        for w, _ in ordered_counter.items():
            if w not in special_tokens:
                i2w[len(w2i)] = w
                w2i[w] = len(w2i)
            else:
                unique_words.append(w)
        if len(w2i) != len(i2w):
            print(unique_words)
            raise ValueError("Mismatch between w2i and i2w mapping")

        # save vocabulary to disk
        vocab = dict(w2i=w2i, i2w=i2w)
        with io.open(vocab_filepath, "wb") as vocab_file:
            jd = json.dumps(vocab, ensure_ascii=False)
            vocab_file.write(jd.encode("utf8", "replace"))

        return vocab

    def _load_vocab(self, vocab_filepath=None):
        """
        Load the vocabulary from a JSON file.

        Parameters:
            vocab_filepath (str): The filepath of the JSON file containing the vocabulary.
            If None, a new vocabulary file will be created.

        Returns:
            tuple: A tuple containing two dictionaries: w2i (word to index) and i2w (index to word).
        """
        if vocab_filepath is not None:
            with open(vocab_filepath, "r") as vocab_file:
                vocab = json.load(vocab_file)
        else:
            new_filepath = os.path.join(self.data_dir, "vocab.json")
            vocab = self._create_vocab(vocab_filepath=new_filepath)
        return (vocab["w2i"], vocab["i2w"])

    def __get_augmented_view__(self, idx, z, change_list):
        """
        Returns an augmented view of the dataset based on the given latent vector `z` and change list.

        Args:
            idx (int): Index of the original view in the dataset.
            z (numpy.ndarray): Latent vector representing the original view.
            change_list (list): List of indices indicating which factors to change in the latent vector.

        Returns:
            view: Augmented view of the dataset.

        """
        z_tilde = np.copy(z)
        for j in change_list:
            if j in self.DISCRETE_FACTORS["image"]:
                z_tilde[j] = (
                    self.LATENT_SPACES["image"][j].uniform(size=1, original=z[j], device="cpu").numpy().flatten()
                )
            else:
                z_tilde[j] = self.LATENT_SPACES["image"][j].uniform(size=1, device="cpu").numpy().flatten()

        return MultiviewDataset.__search_view__(
            z=z_tilde,
            _index=self._index,
            latents=self.z_image.values,
            image_paths=self.image_paths,
            transform=self.transform,
            loader=self.loader,
            augment=True,
            idx_original=idx,
        )

    def __getitem__(self, idx):
        """
        Retrieves the data samples at the given index.

        Args:
            idx (int): Index of the data sample to retrieve.

        Returns:
            dict: A dictionary containing the data samples.
        """

        samples = self.__get_img_text__(idx)
        z_dict = samples["z_image"]
        z_values = np.fromiter(z_dict.values(), dtype=float)

        for k, v in samples.items():
            samples[k] = [v]

        # iterate over number of views and perturb different latents to generate augmented views
        for k in range(len(self.change_lists)):
            index_z_tilde = self.__get_augmented_view__(
                idx=idx,
                z=z_values,
                change_list=self.change_lists[k],
            )[0]
            sample = self.__get_img_text__(index_z_tilde)
            for key, v in sample.items():
                samples[key] += [v]

        # only use the text for the original view
        samples["text"] = [samples["text"][0]]
        samples["z_text"] = [samples["z_text"][0]]

        return samples

    def __get_img_text__(self, idx):
        """
        Get the image, text, and labels for a given index.

        Args:
            idx (int): Index of the sample.

        Returns:
            dict: A dictionary containing the image, text, and labels.
                - "image" (torch.Tensor): The image data.
                - "text" (torch.Tensor): The one-hot encoded text data.
                - "z_image" (dict): The labels for the image.
                - "z_text" (dict): The labels for the text.
        """

        if torch.is_tensor(idx):
            idx = idx.tolist()

        # load image
        img_name = self.image_paths[idx]
        image = pil_loader(img_name)
        if self.transform is not None:
            image = self.transform(image)

        # load text
        words = self.text_in_words[idx]
        words = words + ["{eos}"]
        words = words + ["{pad}" for c in range(self.max_sequence_length - len(words))]
        indices = [self.get_w2i(word) for word in words]
        indices_onehot = torch.nn.functional.one_hot(torch.Tensor(indices).long(), self.vocab_size).float()

        # load labels
        if self.has_labels:
            z_image = {k: v[idx] for k, v in self.labels["z_image"].items()}
            z_text = {k: v[idx] for k, v in self.labels["z_text"].items()}
        else:
            z_image, z_text = None, None

        sample = {
            "image": image,
            "text": indices_onehot,
            "z_image": z_image,
            "z_text": z_text,
        }
        return sample

    def __len__(self):
        """
        Returns the number of samples in the dataset.
        """
        return self.num_samples
