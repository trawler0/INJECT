import os
import os.path
from pathlib import Path
from typing import Any, Callable, cast, Dict, List, Optional, Tuple, Union
import random
from PIL import Image
import json
from torchvision.datasets import VisionDataset

"""
This code was taken from https://github.com/pytorch/vision/blob/main/torchvision/datasets/folder.py
and modified to be used in the INJECT project.
"""


def has_file_allowed_extension(filename: str, extensions: Union[str, Tuple[str, ...]]) -> bool:
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file
        extensions (tuple of strings): extensions to consider (lowercase)

    Returns:
        bool: True if the filename ends with one of given extensions
    """
    return filename.lower().endswith(extensions if isinstance(extensions, str) else tuple(extensions))


def is_image_file(filename: str) -> bool:
    """Checks if a file is an allowed image extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    return has_file_allowed_extension(filename, IMG_EXTENSIONS)


def find_classes(directory: Union[str, Path]) -> Tuple[List[str], Dict[str, int]]:
    """Finds the class folders in a dataset.

    See :class:`DatasetFolder` for details.
    """
    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
    if not classes:
        raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx


def make_dataset(
    directory: Union[str, Path],
    class_to_idx: Optional[Dict[str, int]] = None,
    extensions: Optional[Union[str, Tuple[str, ...]]] = None,
    is_valid_file: Optional[Callable[[str], bool]] = None,
    allow_empty: bool = False,
    seed: int = -1,
    n_shot: int = -1,
    start_shot: int = 0
) -> List[Tuple[str, int]]:
    """Generates a list of samples of a form (path_to_sample, class).

    See :class:`DatasetFolder` for details.

    Note: The class_to_idx parameter is here optional and will use the logic of the ``find_classes`` function
    by default.
    """
    directory = os.path.expanduser(directory)

    if class_to_idx is None:
        _, class_to_idx = find_classes(directory)
    elif not class_to_idx:
        raise ValueError("'class_to_index' must have at least one entry to collect any samples.")

    both_none = extensions is None and is_valid_file is None
    both_something = extensions is not None and is_valid_file is not None
    if both_none or both_something:
        raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")

    if extensions is not None:

        def is_valid_file(x: str) -> bool:
            return has_file_allowed_extension(x, extensions)  # type: ignore[arg-type]

    is_valid_file = cast(Callable[[str], bool], is_valid_file)

    instances = []
    available_classes = set()
    for target_class in sorted(class_to_idx.keys()):
        class_index = class_to_idx[target_class]
        target_dir = os.path.join(directory, target_class)
        if not os.path.isdir(target_dir):
            continue
        for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
            iterator = sorted(fnames)
            if n_shot != -1:
                if seed is not None:
                    random.seed(seed)
                    if len(iterator) > n_shot:
                        iterator = random.sample(iterator, n_shot)
                        iterator = iterator[start_shot:]
                else:
                    iterator = iterator[start_shot:n_shot]  # some authors use the first n_shot images
            for fname in iterator:
                path = os.path.join(root, fname)
                if is_valid_file(path):
                    item = path, class_index
                    instances.append(item)

                    if target_class not in available_classes:
                        available_classes.add(target_class)

    empty_classes = set(class_to_idx.keys()) - available_classes
    if empty_classes and not allow_empty:
        msg = f"Found no valid file for the classes {', '.join(sorted(empty_classes))}. "
        if extensions is not None:
            msg += f"Supported extensions are: {extensions if isinstance(extensions, str) else ', '.join(extensions)}"
        raise FileNotFoundError(msg)

    return instances


class DatasetFolder(VisionDataset):
    """A generic data loader.

    This default directory structure can be customized by overriding the
    :meth:`find_classes` method.

    Args:
        root (str or ``pathlib.Path``): Root directory path.
        loader (callable): A function to load a sample given its path.
        extensions (tuple[string]): A list of allowed extensions.
            both extensions and is_valid_file should not be passed.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.
        is_valid_file (callable, optional): A function that takes path of a file
            and check if the file is a valid file (used to check of corrupt files)
            both extensions and is_valid_file should not be passed.
        allow_empty(bool, optional): If True, empty folders are considered to be valid classes.
            An error is raised on empty folders if False (default).

     Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def __init__(
        self,
        root: Union[str, Path],
        loader: Callable[[str], Any],
        extensions: Optional[Tuple[str, ...]] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
        allow_empty: bool = False,
        seed: int = -1,
        n_shot: int = -1,
        start_shot: int = 0
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        classes, class_to_idx = self.find_classes(self.root)
        samples = self.make_dataset(
            self.root,
            class_to_idx=class_to_idx,
            extensions=extensions,
            is_valid_file=is_valid_file,
            allow_empty=allow_empty,
            seed=seed,
            n_shot=n_shot,
            start_shot=start_shot
        )

        self.loader = loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.idx_to_class = {v: k.replace("_", " ") for k, v in class_to_idx.items()}
        self.idx_to_class = {i: self.idx_to_class[i] for i in range(len(self.class_to_idx))}
        self.samples = samples
        self.targets = [s[1] for s in samples]

    def make_dataset(
        self,
        directory: Union[str, Path],
        class_to_idx: Dict[str, int],
        extensions: Optional[Tuple[str, ...]] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
        allow_empty: bool = False,
        seed: int = -1,
        n_shot: int = -1,
        start_shot: int = 0
    ) -> List[Tuple[str, int]]:
        """Generates a list of samples of a form (path_to_sample, class).

        This can be overridden to e.g. read files from a compressed zip file instead of from the disk.

        Args:
            directory (str): root dataset directory, corresponding to ``self.root``.
            class_to_idx (Dict[str, int]): Dictionary mapping class name to class index.
            extensions (optional): A list of allowed extensions.
                Either extensions or is_valid_file should be passed. Defaults to None.
            is_valid_file (optional): A function that takes path of a file
                and checks if the file is a valid file
                (used to check of corrupt files) both extensions and
                is_valid_file should not be passed. Defaults to None.
            allow_empty(bool, optional): If True, empty folders are considered to be valid classes.
                An error is raised on empty folders if False (default).
            seed (int, optional): Seed for the random number generator. Defaults to -1 (first samples).
            n_shot (int, optional): Number of samples to take from each class. Defaults to -1 (all).
            start_shot: (int, optional): Index of the first sample to take from each class. Defaults to 0.

        Raises:
            ValueError: In case ``class_to_idx`` is empty.
            ValueError: In case ``extensions`` and ``is_valid_file`` are None or both are not None.
            FileNotFoundError: In case no valid file was found for any class.

        Returns:
            List[Tuple[str, int]]: samples of a form (path_to_sample, class)
        """
        if class_to_idx is None:
            # prevent potential bug since make_dataset() would use the class_to_idx logic of the
            # find_classes() function, instead of using that of the find_classes() method, which
            # is potentially overridden and thus could have a different logic.
            raise ValueError("The class_to_idx parameter cannot be None.")
        return make_dataset(
            directory, class_to_idx, extensions=extensions, is_valid_file=is_valid_file, allow_empty=allow_empty,
            seed=seed, n_shot=n_shot, start_shot=start_shot
        )

    def find_classes(self, directory: Union[str, Path]) -> Tuple[List[str], Dict[str, int]]:
        """Find the class folders in a dataset structured as follows::

            directory/
            ├── class_x
            │   ├── xxx.ext
            │   ├── xxy.ext
            │   └── ...
            │       └── xxz.ext
            └── class_y
                ├── 123.ext
                ├── nsdf3.ext
                └── ...
                └── asd932_.ext

        This method can be overridden to only consider
        a subset of classes, or to adapt to a different dataset directory structure.

        Args:
            directory(str): Root directory path, corresponding to ``self.root``

        Raises:
            FileNotFoundError: If ``dir`` has no class folders.

        Returns:
            (Tuple[List[str], Dict[str, int]]): List of all classes and dictionary mapping each class to an index.
        """
        return find_classes(directory)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self) -> int:
        return len(self.samples)

    def save_ids(self, file_path):
        samples = [[os.path.relpath(path, self.root), label] for path, label in self.samples]
        save_tuples_to_json(samples, file_path)


IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp")


def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


def accimage_loader(path: str) -> Any:
    import accimage

    try:
        return accimage.Image(path)
    except OSError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path: str) -> Any:
    from torchvision import get_image_backend

    if get_image_backend() == "accimage":
        return accimage_loader(path)
    else:
        return pil_loader(path)


# save the actual few-shot splits as json to ensure reproducibility and transparency
def save_tuples_to_json(tuples_list, file_path):
    # Convert the list of tuples to a list of lists
    list_of_lists = [list(t) for t in tuples_list]

    # Save the list of lists to a JSON file
    with open(file_path, 'w') as json_file:
        json.dump(list_of_lists, json_file)
    print(f'Saved {len(tuples_list)} tuples to {file_path}')


def load_tuples_from_json(file_path):
    # Load the list of lists from the JSON file
    with open(file_path, 'r') as json_file:
        list_of_lists = json.load(json_file)

    # Convert the list of lists back to a list of tuples
    tuples_list = [tuple(lst) for lst in list_of_lists]
    print(f'Loaded {len(tuples_list)} tuples from {file_path}')
    return tuples_list

class FewShotDataset(DatasetFolder):
    """A generic data loader where the images are arranged in this way by default: ::

        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/[...]/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/[...]/asd932_.png

    This class inherits from :class:`~torchvision.data.DatasetFolder` so
    the same methods can be overridden to customize the dataset.

    Args:
        root (str or ``pathlib.Path``): Root directory path.
        transform (callable, optional): A function/transform that takes in a PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
        is_valid_file (callable, optional): A function that takes path of an Image file
            and check if the file is a valid file (used to check of corrupt files)
        allow_empty(bool, optional): If True, empty folders are considered to be valid classes.
            An error is raised on empty folders if False (default).

     Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        loader: Callable[[str], Any] = default_loader,
        is_valid_file: Optional[Callable[[str], bool]] = None,
        allow_empty: bool = False,
        seed: int = -1,
        n_shot: int = -1,
        start_shot: int = 0,
        mislabel_randomly: float = 0.0
    ):
        super().__init__(
            root,
            loader,
            IMG_EXTENSIONS if is_valid_file is None else None,
            transform=transform,
            target_transform=target_transform,
            is_valid_file=is_valid_file,
            allow_empty=allow_empty,
            seed=seed,
            n_shot=n_shot,
            start_shot=start_shot
        )
        self.imgs = self.samples
        self.random_seed = seed

        if mislabel_randomly > 0:
            random.seed(seed)
            idx = list(self.idx_to_class.keys())
            for i in range(len(self.samples)):
                if random.random() < mislabel_randomly:
                    self.samples[i] = (self.samples[i][0], random.choice(idx))
                    self.targets[i] = self.samples[i][1]


class FewShotSplitDataset(FewShotDataset):

    def __init__(
            self,
            root: str,
            split: str,
            image_dir: str,
            split_file: str,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            loader: Callable[[str], Any] = default_loader,
            is_valid_file: Optional[Callable[[str], bool]] = None,
            allow_empty: bool = False,
            seed: int = -1,
            n_shot: int = -1,
            start_shot: int = 0,
            mislabel_randomly: float = 0.0
    ):
        self.split_file = os.path.join(root, split_file)
        assert split in ["train", "val", "test"]
        self.split = split
        self.image_dir = image_dir
        super().__init__(
            root,
            transform=transform,
            target_transform=target_transform,
            loader=loader,
            is_valid_file=is_valid_file,
            allow_empty=allow_empty,
            seed=seed,
            n_shot=n_shot,
            start_shot=start_shot,
            mislabel_randomly=mislabel_randomly
        )

    def find_classes(self, directory: Union[str, Path]) -> Tuple[List[str], Dict[str, int]]:
        classes = []
        class_to_idx = {}
        with open(self.split_file, "r") as f:
            split_file = json.load(f)[self.split]
            for _, label, class_name in split_file:
                if class_name not in classes:
                    classes.append(class_name)
                    class_to_idx[class_name] = label
        return classes, class_to_idx

    def make_dataset(
        self,
        directory: Union[str, Path],
        class_to_idx: Dict[str, int],
        extensions: Optional[Tuple[str, ...]] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
        allow_empty: bool = False,
        seed: int = -1,
        n_shot: int = -1,
        start_shot: int = 0
    ) -> List[Tuple[str, int]]:

        with open(self.split_file, 'r') as json_file:
            samples_preliminary = json.load(json_file)[self.split]

        image_folder = os.path.join(self.root, self.image_dir)
        samples = []
        count = {}
        if seed != -1:
            random.seed(seed)
            random.shuffle(samples_preliminary)
        for im_name, label, class_name in samples_preliminary:
            if class_name not in count:
                count[class_name] = 0
            if count[class_name] == n_shot:
                continue
            if count[class_name] >= start_shot:
                im_path = os.path.join(image_folder, im_name)
                samples.append((im_path, label))
            count[class_name] += 1
        return samples

class LoadFewShotDataset(VisionDataset):

    def __init__(self, root: str, split_file: Union[str, Path], transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None, loader: Callable[[str], Any] = default_loader):
        super().__init__(root, transform, target_transform)
        self.samples = load_tuples_from_json(split_file)
        self.samples = [(os.path.join(root, path), label) for path, label in self.samples]
        self.imgs = self.samples
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        path, target = self.samples[index]
        path = os.path.join(self.root, path)
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self) -> int:
        return len(self.samples)


def only_split_train(split, kwargs):

    if split != "train":
        kwargs["n_shot"] = -1

    return kwargs

class Registry:
    def __init__(self):
        self._registry = {}

    def register(self, name):
        def wrapper(cls):
            """Decorator to register a class"""
            self._registry[name] = cls
            return cls
        return wrapper

    def get(self, name):
        return self._registry.get(name)

    def __str__(self):
        return str(self._registry)















