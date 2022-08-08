import numpy as np
from torchvision import transforms


class PoisonedDataset():
    '''
    trainset - full training set containing all indices
    indices  - subset of training set
    poison_instances - list of tuples of poison examples
                       and their respective labels like
                       [(x_0, y_0), (x_1, y_1) ...]
                       this must correspond to poison indices
    poison_indices - list of indices which are poisoned
    transform - transformation to apply to each image
    return_index - whether to return original index into trainset
    '''
    def __init__(self, trainset, indices, poison_instances=[], poison_indices=[], transform=None, return_index=False, size=None):
        if len(poison_indices) or len(poison_instances):
            assert len(poison_indices) == len(poison_instances)
        self.trainset = trainset 
        self.indices = indices
        self.transform = transform
        self.poisoned_label = (
            None if len(poison_instances) == 0 else poison_instances[0][1]
        )
        self.return_index = return_index
        if size is None:
            size = len(indices)
        if size < len(indices):
            self.find_indices(size, poison_indices, poison_instances)

        # Set up new indexing
        if len(poison_instances) > 0:

            poison_mask = np.isin(poison_indices, self.indices)
            poison_mask = np.nonzero(poison_mask)[0]
            self.poison_map = {
                int(poison_indices[i]): poison_instances[i]
                for i in poison_mask
            }

            clean_mask = np.isin(self.indices, poison_indices, invert=True)
            self.clean_indices = self.indices[clean_mask]
        else:
            self.poison_map = {}
            self.clean_indices = self.indices

        self.to_pil = transforms.ToPILImage()


    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        index = self.indices[index]
        if index in self.poison_map:
            img, label = self.poison_map[index]
            p = 1
        else:
            img, label = self.trainset[index]
            p = 0
        if self.transform is not None:
            try:
                img = self.transform(img)
            except:
                img = self.to_pil(np.uint8(img))
                img = self.transform(img)
        if self.return_index:
            return img, label, p, index
        else:
            return img, label, p  # 0 for unpoisoned, 1 for poisoned

    def find_indices(self, size, poison_indices, poison_instances):
        good_idx = np.array([])
        batch_tar = np.array(self.trainset.targets)
        num_classes = len(set(batch_tar))
        num_per_class = int(size / num_classes)
        for label in range(num_classes):
            all_idx_for_this_class = np.where(batch_tar == label)[0]
            all_idx_for_this_class = np.setdiff1d(
                all_idx_for_this_class, poison_indices
            )
            this_class_idx = all_idx_for_this_class[:num_per_class]
            if label == self.poisoned_label and len(poison_instances) > 0:
                num_clean = num_per_class - len(poison_instances)
                this_class_idx = this_class_idx[:num_clean]
                this_class_idx = np.concatenate((this_class_idx, poison_indices))
            good_idx = np.concatenate((good_idx, this_class_idx))
        good_idx = good_idx.astype(int)
        self.indices = good_idx[np.isin(good_idx, self.indices)]
