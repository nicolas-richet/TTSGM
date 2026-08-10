import torch
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
import json
import os
import numpy as np
from PIL import Image
import random
from utils.cifar_new import CIFARNew 
from utils.tiny_imagenet import TinyImageNetDataset
from utils.tiny_imagenet_c import TinyImageNetCDataset
from utils.datautils import GenericImageDataset

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

def _convert_image_to_rgb(image):
    return image.convert("RGB")

import clip
#_, CLIP_TRANSFORMS = clip.load('ViT-B/32')
CLIP_TRANSFORMS = transforms.Compose([
                                transforms.Resize(224, interpolation=BICUBIC),
                                transforms.CenterCrop(224),
                                _convert_image_to_rgb,
                                transforms.ToTensor(),
                                transforms.Normalize((0.48145466, 0.4578275, 0.40821073), 
                                                     (0.26862954, 0.26130258, 0.27577711)),
                                ])


CIFAR_COMMON_CORRUPTIONS = ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur',
                            'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog', 'brightness', 'contrast', 
                            'elastic_transform', 'pixelate', 'jpeg_compression']

SPLIT_FILE_PATHS = {
    'caltech101': "/Caltech101/split_zhou_Caltech101.json",
    'dtd': '/DTD/split_zhou_DescribableTextures.json',
    'eurosat': "/eurosat/split_zhou_EuroSAT.json",
    'fgvc_aircraft': "/fgvc_aircraft", 
    'flower102': "/Flower102/split_zhou_OxfordFlowers.json", 
    'food101': "/Food101/split_zhou_Food101.json", 
    'oxfordpets': "/OxfordPets/split_zhou_OxfordPets.json", 
    'stanfordcars': "/StanfordCars/split_zhou_StanfordCars.json", 
    'sun397': "/SUN397/split_zhou_SUN397.json", 
    'ucf101': "/UCF101/split_zhou_UCF101.json"
}


def prepare_data(dataset, data_dir, corruption, batch_size=128, num_workers=1, test=False, noise_ratio=0, severity=5):

    """
    Prepare the specified dataset.

    Parameters:
    ----------
    dataset : str
        The name of the dataset to prepare. Should be one of 'cifar10', 'cifar100', 'tiny-imagenet', 'visda', 'PACS', 'office_home', 'VLCS'.
    data_dir : str
        The root directory where the dataset is stored.
    corruption : str
        The type of corruption to apply to the dataset, if applicable. 
        Only used for 'cifar10', 'cifar100', and 'tiny-imagenet'.
    batch_size : int, optional
        The number of samples per batch to load. Default is 128.
    num_workers : int, optional
        The number of subprocesses to use for data loading. Default is 1.

    Returns:
    -------
    tuple
        A tuple containing:
        - loader (torch.utils.data.DataLoader): DataLoader for the prepared dataset.
        - dataset (torchvision.datasets or ImageFolder): The prepared dataset.

    """


    if dataset == 'cifar10':
        loader, classes = prepare_cifar10_data(data_dir, corruption, batch_size=batch_size, num_workers=num_workers, noise_ratio=noise_ratio, level=severity)

    elif dataset == 'cifar100':
        loader, classes = prepare_cifar100_data(data_dir, corruption, batch_size=batch_size, num_workers=num_workers, level=severity)
    
    elif dataset == 'tiny-imagenet':
        loader, classes = prepare_tinyimagenet(data_dir, corruption=corruption, batch_size=batch_size, num_workers=num_workers)

    elif dataset in  ["visda", "PACS", "office_home", "VLCS", "dtd", "imagenetr", "imagenetv2", "caltech101", "dtd", "eurosat", 'fgvc_aircraft', 'flower102', 'food101', 'oxfordpets', 'stanfordcars', 'sun397', 'ucf101']:
        if test:
            train_loader, test_loader, classes = prepare_imagefolder_data(data_dir, dataset, batch_size=batch_size, num_workers=num_workers, test=test)
            return train_loader, test_loader, classes
        else:
            loader, classes = prepare_imagefolder_data(data_dir, dataset, batch_size=batch_size, num_workers=num_workers)

    else:
        raise Exception(f'Dataset {dataset} not found/implemented!')
    
    return loader, classes





def prepare_cifar10_data(data_dir, corruption, batch_size=128, level=5, size=10000, num_workers=1, noise_ratio=0): 
    """
    A function to prepare different versions of CIFAR-10 dataset.

    Parameters:
    ----------
    data_dir : str
        The root directory where the dataset is stored.
    corruption : str
        The type of corruption to apply to the dataset.
        Should be 'original', one of CIFAR_COMMON_CORRUPTIONS, or 'cifar_new'.
    batch_size : int, optional
        The number of samples per batch to load. Default is 128.
    level : int, optional
        The severity level of corruption for CIFAR-10-C. Default is 5.
    size : int, optional
        The number of images to select per corruption level for CIFAR-10-C. Default is 10000.
    num_workers : int, optional
        The number of subprocesses to use for data loading. Default is 1.

    Returns:
    -------
    tuple
        A tuple containing:
        - loader (torch.utils.data.DataLoader): DataLoader for the prepared dataset.
        - dataset (torchvision.datasets): The prepared dataset.

    """
    if corruption == 'original':
        dataset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=False, transform=CLIP_TRANSFORMS)
        print(f"The original CIFAR-10 dataset is selected, number of images: {len(dataset)}!\ndatadir: {data_dir}")
    
    elif corruption in CIFAR_COMMON_CORRUPTIONS:
        dataset_raw = np.load(data_dir + '/CIFAR-10-C/%s.npy' % (corruption))
        dataset_raw = dataset_raw[(level - 1)*size: level*size]
        dataset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=False, transform=CLIP_TRANSFORMS)
        dataset.data = dataset_raw
        print(f"The CIFAR-10-C dataset with '{corruption}' corruption is selected, number of images: {len(dataset)}!\ndatadir: {data_dir + '/CIFAR-10-C'}")

        if noise_ratio > 0:
            dataset = NoisyLabelDataset(dataset, noise_ratio, num_classes=10)
            print(f'Noise ratio = {noise_ratio}')

    elif corruption == 'cifar_new':
        dataset = CIFARNew(root=data_dir + '/CIFAR-10.1/', transform=CLIP_TRANSFORMS)
        print(f"The CIFAR-10.1 (new CIFAR10 test data) dataset is selected, number of images: {len(dataset)}!\ndatadir: {data_dir + '/CIFAR-10.1/'}")

    else:
        raise Exception(f'Corruption {corruption} not found!')

    g = torch.Generator()
    g.manual_seed(0)  # fixed global seed

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=g
    )
    
    classes = dataset.classes
    return loader, classes



def prepare_cifar100_data(data_dir, corruption, batch_size=128, level=5, size=10000, num_workers=1): 
    """
    A function to prepare different versions of CIFAR-100 dataset.

    Parameters:
    ----------
    data_dir : str
        The root directory where the dataset is stored.
    corruption : str
        The type of corruption to apply to the dataset. 
        Should be 'original' or one of CIFAR_COMMON_CORRUPTIONS.
    batch_size : int, optional
        The number of samples per batch to load. Default is 128.
    level : int, optional
        The severity level of corruption for CIFAR-100-C. Default is 5.
    size : int, optional
        The number of images to select per corruption level for CIFAR-100-C. Default is 10000.
    num_workers : int, optional
        The number of subprocesses to use for data loading. Default is 1.

    Returns:
    -------
    tuple
        A tuple containing:
        - loader (torch.utils.data.DataLoader): DataLoader for the prepared dataset.
        - dataset (torchvision.datasets.CIFAR100): The prepared dataset.

    """

    
    if corruption == 'original':
        dataset = torchvision.datasets.CIFAR100(root=data_dir, train=False, download=False, transform=CLIP_TRANSFORMS)
        print(f"The original CIFAR-100 dataset is selected, number of images: {len(dataset)}!\ndatadir: {data_dir}")

    elif corruption in CIFAR_COMMON_CORRUPTIONS:
        dataset_raw = np.load(data_dir + '/CIFAR-100-C/%s.npy' % (corruption))
        dataset_raw = dataset_raw[(level - 1)*size: level*size]
        dataset = torchvision.datasets.CIFAR100(root=data_dir, train=False, download=False, transform=CLIP_TRANSFORMS)
        dataset.data = dataset_raw
        print(f"The CIFAR-100-C dataset with '{corruption}' corruption is selected, number of images: {len(dataset)}!\ndatadir: {data_dir}")
    
    else:
        raise Exception(f'Corruption {corruption} not found!')

    g = torch.Generator()
    g.manual_seed(0)  # fixed global seed

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=g
    )
    classes = dataset.classes
    return loader, classes



def prepare_tinyimagenet(data_dir, corruption, batch_size=128, level=5, num_workers=1):
    """
    Prepare different versions of the TinyImageNet dataset.

    Parameters:
    ----------
    data_dir : str
        The root directory where the dataset is stored.
    corruption : str
        The type of corruption to apply to the dataset.
        Should be 'original' or one of CIFAR_COMMON_CORRUPTIONS.
    batch_size : int, optional
        The number of samples per batch to load. Default is 128.
    level : int, optional
        The severity level of corruption for Tiny-ImageNet-C. Default is 5.
    num_workers : int, optional
        The number of subprocesses to use for data loading. Default is 1.

    Returns:
    -------
    tuple
        A tuple containing:
        - loader (torch.utils.data.DataLoader): DataLoader for the prepared dataset.
        - dataset (TinyImageNetDataset or TinyImageNetCDataset): The prepared dataset.

    
    """
    if corruption == 'original':
        dataset = TinyImageNetDataset(data_dir + '/tiny-imagenet-200/', mode='val', transform=CLIP_TRANSFORMS)
        print(f"The original TinyImageNetDataset dataset is selected, number of images: {len(dataset)}!\ndatadir: {data_dir}")
    
    elif corruption in CIFAR_COMMON_CORRUPTIONS:
        dataset = TinyImageNetCDataset(data_dir + '/Tiny-ImageNet-C/', corruption=corruption, level=level, transform=CLIP_TRANSFORMS)
        print(f"The TinyImageNetDataset dataset with '{corruption}' corruption is selected, number of images: {len(dataset)}!\ndatadir: {data_dir}")

    g = torch.Generator()
    g.manual_seed(0)  # fixed global seed

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=g
    )    
    classes = dataset.classes
    return loader, classes


def prepare_imagefolder_data(data_dir, name, batch_size=128, num_workers=1, test=False, test_ratio=0.2, seed=42):
    """
    Prepare an image dataset from a directory using ImageFolder.
    If test=True, split the dataset into train and test sets with a fixed seed.

    Parameters:
    ----------
    data_dir : str
        The root directory where the dataset is stored.
    name : str
        The name of the dataset.
    batch_size : int, optional
        The number of samples per batch to load. Default is 128.
    num_workers : int, optional
        The number of subprocesses to use for data loading. Default is 1.
    test : bool, optional
        If True, split dataset into train/test sets and return both. Default is False.
    test_ratio : float, optional
        Proportion of the dataset to allocate for testing. Default is 0.2.
    seed : int, optional
        Random seed for reproducible splits. Default is 42.

    Returns:
    -------
    If test=False:
        tuple: (loader, classes)
            loader (torch.utils.data.DataLoader): DataLoader for the dataset.
            classes (list): List of class names.
    If test=True:
        tuple: (train_loader, test_loader, classes)
            train_loader (torch.utils.data.DataLoader): DataLoader for the train set.
            test_loader (torch.utils.data.DataLoader): DataLoader for the test set.
            classes (list): List of class names.
    """
    drop_last = False
    
    if name in SPLIT_FILE_PATHS.keys():
        if name == 'fgvc_aircraft':
            split_data = []

            # --- Build mapping from class names ---
            classnames = []
            root = "/".join(data_dir.split('/')[:-1])
            with open(os.path.join(root, 'variants.txt'), 'r') as f:
                lines = f.readlines()
                for line in lines:
                    classnames.append(line.strip())
            cname2lab = {c: i for i, c in enumerate(classnames)}

            # --- Parse test split file ---
            with open(os.path.join(root, 'images_variant_test.txt'), 'r') as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip().split(' ')
                    imname = line[0] + '.jpg'
                    classname = ' '.join(line[1:])
                    label = cname2lab[classname]
                    split_data.append((imname, label, classname))

            # --- Directly initialize dataset with split_data ---
            dataset = GenericImageDataset(
                data_root=data_dir,
                transform=CLIP_TRANSFORMS,
                split_data=split_data
            )
        else:
            split_path = SPLIT_FILE_PATHS[name]

            # --- Initialize dataset directly from JSON split ---
            dataset = GenericImageDataset(
                data_root=data_dir,
                transform=CLIP_TRANSFORMS,
                split_path=split_path,
                split_name="test"
            )

    else:
        # No split file → load full dataset (works for ImageFolder or flat+json)
        dataset = GenericImageDataset(
            data_root=data_dir,
            transform=CLIP_TRANSFORMS
        )
        print(f"dataset_len : {len(dataset)}")
        drop_last=False
        
    if name == 'imagenetv2':
        imagenet_classes = json.load(open(os.path.join(data_dir, "imagenet_class_index.json")))
        imagenet_class_names = {key:imagenet_classes[key][1] for key in imagenet_classes.keys()}
        classes = [imagenet_class_names[x] for x in classes]
    
    if not test:
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                            num_workers=num_workers, pin_memory=True, drop_last=drop_last, shuffle=True)
        print(f"The original {name} dataset is selected, number of images: {len(dataset)}!\ndatadir: {data_dir}")
        classes = dataset.classes
        return loader, classes
    else:
        # compute split sizes
        test_size = int(len(dataset) * test_ratio)
        train_size = len(dataset) - test_size

        # fixed seed generator
        g = torch.Generator().manual_seed(seed)

        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size], generator=g)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                  num_workers=num_workers, pin_memory=True, drop_last=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                 num_workers=num_workers, pin_memory=True, drop_last=False)

        print(f"{name} dataset split: {train_size} train / {test_size} test\nDatadir: {data_dir}")
        return train_loader, test_loader, classes


def seed_worker(i=0):
    # Ensure each worker has a reproducible, but different, seed
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    
class NoisyLabelDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, noise_ratio, num_classes, seed=0):
        self.base = base_dataset
        self.noise_ratio = noise_ratio
        self.num_classes = num_classes
        self.classes = base_dataset.classes
        rng = np.random.default_rng(seed)
        self.clean_labels = np.array(base_dataset.targets)

        n = len(self.clean_labels)
        n_noisy = int(noise_ratio * n)
        self.noisy_idx = rng.choice(n, n_noisy, replace=False)

        self.noisy_labels = self.clean_labels.copy()
        for i in self.noisy_idx:
            y = self.clean_labels[i]
            new_y = rng.integers(0, num_classes - 1)
            if new_y >= y:
                new_y += 1
            self.noisy_labels[i] = new_y

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        x, _ = self.base[idx]
        y = self.noisy_labels[idx]
        return x, y