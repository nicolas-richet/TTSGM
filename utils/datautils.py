import torch
import os
import json
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.datasets.folder import default_loader
class Augmenter(object):
    def __init__(self, transform, n_views):
        self.transform= transform
        self.n_views = n_views
    
    def augment_batch(self, inputs, labels):
        imgs = [inputs]
        for _ in range(self.n_views-1):
            imgs.append(self.transform(inputs))
        labels = labels.unsqueeze(0).repeat([self.n_views,1])
        labels = labels.flatten()
        imgs = torch.stack(imgs, dim=0)
        imgs = imgs.flatten(end_dim=1)
        return imgs, labels

class GenericImageDataset(Dataset):
    def __init__(self, data_root, transform=None, split_path=None, split_name="test", split_data=None):
        """
        Args:
            data_root (str): root folder of images
            transform: torchvision transform
            split_path (str): optional path to JSON split file
            split_name (str): which split to use in JSON (e.g., "train", "test")
            split_data (list): optional split data directly passed (list of [rel_path, class_id, class_name])
        """
        self.data_root = data_root
        self.transform = transform
        self.loader = default_loader

        # --- Case 1: split_data is given directly ---
        if split_data is not None:
            self._init_from_split(split_data)
            return

        # --- Case 2: try to detect folder structure ---
        has_subfolders = any(
            os.path.isdir(os.path.join(data_root, d)) for d in os.listdir(data_root)
        )

        if has_subfolders:
            base_dataset = datasets.ImageFolder(root=data_root, transform=transform)
            self.classes = base_dataset.classes
            self.class_to_idx = base_dataset.class_to_idx
            all_samples = base_dataset.samples
        else:
            if split_path is None:
                raise ValueError("Flat folder detected but no JSON split file provided.")
            all_samples = None

        # --- Case 3: JSON split filtering ---
        if split_path is not None:
            with open(split_path, "r") as f:
                split_data = json.load(f)[split_name]
            self._init_from_split(split_data)
        else:
            # No JSON, use full ImageFolder samples
            if not has_subfolders:
                raise ValueError("Flat dataset requires a split JSON file.")
            self.samples = all_samples
            self.targets = [s[1] for s in self.samples]

    def _init_from_split(self, split_data):
        """Initialize dataset directly from a JSON-style split list"""
        # JSON entries: [rel_path, class_id, class_name]
        split_dict = {entry[0]: (entry[1], entry[2]) for entry in split_data}
        id_to_class = {entry[1]: entry[2] for entry in split_data}

        self.classes = [name for _, name in sorted(id_to_class.items())]
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

        self.samples = []
        for rel_path, (label_id, _) in split_dict.items():
            abs_path = os.path.join(self.data_root, rel_path)
            self.samples.append((abs_path, label_id))

        self.targets = [s[1] for s in self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, target = self.samples[idx]
        img = self.loader(path)
        if self.transform:
            img = self.transform(img)
        return img, target