from PIL import Image
import numpy as np
import os
import json
from collections import defaultdict
from torch.utils.data import Dataset

from tqdm.autonotebook import tqdm


def _add_channels(img,):
  while len(img.shape) < 3:  # third axis is the channels
    img = np.expand_dims(img, axis=-1)
  while(img.shape[-1]) < 3:
    img = np.concatenate([img, img[:, :, -1:]], axis=-1)
  return img

class TinyImageNetPaths:
    def __init__(self, root_dir, corruption='original', level=5):
        val_path = os.path.join(root_dir, corruption, str(level))

        label_map_path = os.path.join(root_dir, "imagenet_class_index.json")
        with open(label_map_path, 'r') as f:
            label_dict = json.load(f)

        # Original mappings from JSON
        all_ids = [label_dict[k][0] for k in sorted(label_dict.keys(), key=int)]   # wnids in order
        all_names = [label_dict[k][1] for k in sorted(label_dict.keys(), key=int)] # human names

        corrupt_nids = os.listdir(val_path)

        # Keep only classes actually present in TinyImageNet-C
        self.ids = [nid for nid in all_ids if nid in corrupt_nids]
        self.classes = [label_dict[k][1] for k in sorted(label_dict.keys(), key=int) if label_dict[k][0] in self.ids]


        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        self.nid_to_words = {nid: [name] for nid, name in zip(self.ids, self.classes)}

        self._make_paths(val_path, corrupt_nids)

    def _make_paths(self, corrupt_path, corrupt_nids):
        self.paths = {'corrupt': []}

        for nid in corrupt_nids:
            if nid not in self.ids:
                continue
            label_id = self.ids.index(nid)
            path = os.path.join(corrupt_path, nid)
            for imgname in os.listdir(path):
                fname = os.path.join(path, imgname)
                self.paths['corrupt'].append((fname, label_id, nid))

class TinyImageNetCDataset(Dataset):
    def __init__(self, root_dir, mode='corrupt',
                 transform=None, max_samples=None,
                 corruption='snow', level=5):

        tinp = TinyImageNetPaths(root_dir,
                                 corruption=corruption, level=level)

        self.mode = mode
        self.label_idx = 1  # from [img_path, id, nid]
        self.transform = transform

        self.IMAGE_SHAPE = (64, 64, 3)

        self.samples = tinp.paths[mode]
        self.samples_num = len(self.samples)

        # Add ImageFolder-like attributes
        self.classes = tinp.classes
        self.class_to_idx = tinp.class_to_idx

        if max_samples is not None:
            self.samples_num = min(max_samples, self.samples_num)
            self.samples = np.random.permutation(self.samples)[:self.samples_num]

    def __len__(self):
        return self.samples_num

    def __getitem__(self, idx):
        s = self.samples[idx]
        img = Image.open(s[0])
        img_array = np.array(img)

        if img_array.ndim < 3 or img_array.shape[-1] < 3:
            img_array = _add_channels(img_array)
            img = Image.fromarray(img_array)

        lbl = None if self.mode == 'test' else s[self.label_idx]

        if self.transform:
            img = self.transform(img)

        return img, lbl
