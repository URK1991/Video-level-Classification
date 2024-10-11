import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.utils.class_weight import compute_class_weight

class TrainDataset(Dataset):
    def __init__(self, X, Y, mask_pct=0.00):
        X = np.vstack(list(X))
        Y = np.squeeze(np.vstack(list(Y)))
        
        self.class_weights = torch.Tensor(compute_class_weight(class_weight='balanced', 
                                                                classes=np.unique(Y), y=Y))

        self.X = torch.Tensor(list(X))
        self.Y = nn.functional.one_hot(torch.Tensor(list(Y)).type(torch.int64), num_classes=4)
        self.mask_pct = mask_pct

    def __random_mask(self, index):
        import random
        X1 = self.X[index]
        if self.mask_pct == 0:
            return X1

        n_frames = len(X1)
        f_shape = X1[0].shape

        pix_sums = X1.sum(axis=1)
        pad_start_idx = n_frames - next((i for i, x in enumerate(reversed(pix_sums)) if x), None)

        mask_idxs = random.sample(range(0, pad_start_idx), int(pad_start_idx * self.mask_pct))
        for idx in mask_idxs:
            X1[idx] = torch.zeros(f_shape)

        return X1

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        X1 = self.__random_mask(index)
        return X1, self.Y[index]
